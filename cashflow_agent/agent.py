"""The agentic loop: model proposes tool calls, we execute them, repeat."""
from __future__ import annotations

from dataclasses import dataclass, field

from cashflow_agent.llm import Backend
from cashflow_agent.policy import POLICY_TEXT
from cashflow_agent.synth import CATEGORIES
from cashflow_agent.tools import TOOLS, RunState, ToolError, tool_result_block

SYSTEM_PROMPT = f"""You are a small-business underwriting analyst agent. You review one business bank \
statement at a time using the tools provided, then submit an underwriting memo.

Workflow:
1. Read the overview, then page through every transaction with get_transactions.
2. Classify every transaction with classify_transactions (you can batch a whole page per call). \
Categories: {", ".join(CATEGORIES)}. "mca_remittance" = daily/weekly merchant-cash-advance debits; \
"card_revenue" = card-processor settlements; "customer_payment" = other revenue credits (ACH, wires, \
deposits); "transfer" = movements between the owner's own accounts or to/from the owner. Give an honest \
confidence; anything below 0.8 is sent to a human reviewer, which is the right outcome when the \
descriptor is ambiguous.
3. Call run_anomaly_checks. Its candidates are recall-oriented and can be false positives: read the \
evidence and the descriptors, and confirm only real signals with flag_risk. For example, invoice \
payments that happen to fall in a cash-structuring amount band are not structuring.
4. Call compute_cashflow_features. Never compute metrics yourself; quote the tool's numbers exactly.
5. Apply the credit policy and call submit_underwriting_memo once.

{POLICY_TEXT}

Everything you produce is reviewed by a human credit officer before any decision is made."""


@dataclass
class AgentResult:
    statement_id: str
    status: str                      # completed | refused | truncated | turn_limit | incomplete
    state: RunState
    turns: int
    usage: dict = field(default_factory=dict)
    tool_errors: int = 0


def run_agent(statement: dict, backend: Backend, max_turns: int = 40) -> AgentResult:
    state = RunState(statement)
    messages: list[dict] = [{"role": "user", "content": (
        f"Review statement {statement['statement_id']} and submit an underwriting memo.")}]
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    status, errors = "turn_limit", 0
    turns = 0
    for turns in range(1, max_turns + 1):
        turn = backend.step(SYSTEM_PROMPT, TOOLS, messages)
        for k, v in turn.usage.items():
            usage[k] = usage.get(k, 0) + v
        messages.append({"role": "assistant", "content": turn.raw_content})
        if turn.stop_reason == "refusal":
            status = "refused"
            break
        if turn.stop_reason == "max_tokens":
            status = "truncated"
            break
        uses = [b for b in turn.blocks if b["type"] == "tool_use"]
        if not uses:
            status = "completed" if state.memo else "incomplete"
            break
        results = []
        for u in uses:  # all results go back in ONE user message
            try:
                results.append(tool_result_block(u["id"], state.execute(u["name"], u["input"])))
            except (ToolError, KeyError, TypeError, ValueError) as e:
                errors += 1
                results.append(tool_result_block(u["id"], f"error: {e}", is_error=True))
        messages.append({"role": "user", "content": results})
    return AgentResult(statement["statement_id"], status, state, turns, usage, errors)
