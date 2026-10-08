"""LLM backends behind one interface.

* ClaudeBackend - the Anthropic Messages API with client-side tool use
  (default model claude-haiku-5-5 at low effort, for cost).
* MockBackend   - a fully offline, deterministic scripted agent that speaks the
  same tool-use protocol (tool_use blocks in, tool_result blocks out). It uses
  keyword rules to classify and accepts every detector candidate. It exists so
  the whole pipeline, review queue and eval harness run with no API key, and it
  doubles as a transparent rules baseline the live model has to beat.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from cashflow_agent.policy import FRAUD_SIGNALS, recommend


@dataclass
class Turn:
    raw_content: Any                 # what gets appended to history as the assistant turn
    blocks: list[dict]               # normalised: {"type": "text"|"tool_use", ...}
    stop_reason: str
    usage: dict = field(default_factory=dict)


class Backend(Protocol):
    name: str

    def step(self, system: str, tools: list[dict], messages: list[dict]) -> Turn: ...


# --------------------------------------------------------------------------- Claude
class ClaudeBackend:
    def __init__(self, model: str = "claude-haiku-5-5", effort: str = "low", max_tokens: int = 16000,
                 client=None):
        import anthropic

        self.client = client or anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / `ant auth login`
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.name = f"claude:{model}"

    def step(self, system: str, tools: list[dict], messages: list[dict]) -> Turn:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=tools,
            messages=messages,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},  # system + tools + growing history are a stable prefix
        )
        blocks = []
        for b in resp.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        u = resp.usage
        usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                 "cache_read_input_tokens": u.cache_read_input_tokens or 0,
                 "cache_creation_input_tokens": u.cache_creation_input_tokens or 0}
        return Turn(raw_content=resp.content, blocks=blocks, stop_reason=resp.stop_reason, usage=usage)


# --------------------------------------------------------------------------- Mock
RULES: list[tuple[str, str, float]] = [
    (r"\bNSF\b|OVERDRAFT|RETURNED ITEM", "nsf_fee", 0.97),
    (r"PAYROLL|WAGE", "payroll", 0.95),
    (r"\bLEASE\b|\bRENT\b|PROPERT", "rent", 0.92),
    (r"FUNDING|\bCAP ACH\b|\bCAP RECV\b|ADVANCE|KALAMATA", "mca_remittance", 0.85),
    (r"\bLOAN\b|FINANCE PMT", "loan_repayment", 0.92),
    (r"SQUARE|STRIPE|TOAST|CLOVER|SHOPIFY|SETTLEMENT|PAYOUT", "card_revenue", 0.95),
    (r"\bINS\b|INSURANCE|\bPREM\b", "insurance", 0.90),
    (r"\bIRS\b|EFTPS|\bTAX\b", "tax_payment", 0.95),
    (r"ELECTRIC|\bGAS\b|VERIZON|COMCAST|WASTE|WATER", "utilities", 0.93),
    (r"VENDOR|SYSCO|DEPOT|ULINE|GRAINGER|PAYABLE|FOODS", "supplier_payment", 0.90),
    (r"TRANSFER|ZELLE|XFER", "transfer", 0.90),
    (r"RECEIVABLE|\bINV\b|DEPOSIT|WIRE IN|CONTRACT", "customer_payment", 0.85),
]


def rule_classify(desc: str, amount: float) -> tuple[str, float]:
    d = desc.upper()
    for pat, cat, conf in RULES:
        if re.search(pat, d):
            return cat, conf
    # no signal in the descriptor: guess from direction, low confidence
    return ("customer_payment", 0.45) if amount > 0 else ("supplier_payment", 0.40)


class MockBackend:
    """Deterministic scripted agent. Reads the latest tool_result(s) from the
    conversation and emits the next tool_use, exactly like a model would."""

    name = "mock:rules-v1"

    def __init__(self) -> None:
        self._n = 0

    def _use(self, name: str, inp: dict) -> dict:
        self._n += 1
        return {"type": "tool_use", "id": f"toolu_mock_{self._n:04d}", "name": name, "input": inp}

    def _turn(self, blocks: list[dict], stop: str = "tool_use") -> Turn:
        return Turn(raw_content=blocks, blocks=blocks, stop_reason=stop,
                    usage={"input_tokens": 0, "output_tokens": 0})

    def step(self, system: str, tools: list[dict], messages: list[dict]) -> Turn:
        last = messages[-1]
        if isinstance(last["content"], str):  # first turn
            return self._turn([{"type": "text", "text": "Starting review."},
                               self._use("get_statement_overview", {})])
        prev_calls = {b["id"]: b for b in messages[-2]["content"] if b.get("type") == "tool_use"}
        results = [(prev_calls[r["tool_use_id"]]["name"], json.loads(r["content"]) if not r.get("is_error") else
                    {"error": r["content"]}) for r in last["content"]]
        name, out = results[0]

        if name == "get_statement_overview":
            return self._turn([self._use("get_transactions", {"offset": 0})])
        if name == "get_transactions":
            items = []
            for t in out["transactions"]:
                cat, conf = rule_classify(t["description"], t["amount"])
                items.append({"txn_id": t["txn_id"], "category": cat, "confidence": conf})
            blocks = [self._use("classify_transactions", {"items": items})]
            self._next_offset = out["next_offset"]
            return self._turn(blocks)
        if name == "classify_transactions":
            if self._next_offset is not None:
                return self._turn([self._use("get_transactions", {"offset": self._next_offset})])
            return self._turn([self._use("run_anomaly_checks", {})])
        if name == "run_anomaly_checks":
            self._flagged = set()
            uses = []
            for c in out["candidates"]:
                sev = "high" if c["type"] in FRAUD_SIGNALS else "medium"
                self._flagged.add(c["type"])
                uses.append(self._use("flag_risk", {"signal_type": c["type"], "txn_ids": c["txn_ids"],
                                                    "severity": sev, "evidence": c["evidence"][:500]}))
            return self._turn(uses or [self._use("compute_cashflow_features", {})])
        if name == "flag_risk":
            return self._turn([self._use("compute_cashflow_features", {})])
        if name == "compute_cashflow_features":
            f = out
            rec = recommend(self._flagged, f)
            dscr = "n/a (no debt service)" if f["dscr_proxy"] is None else f"{f['dscr_proxy']}"
            summary = (f"Average daily balance ${f['avg_daily_balance']:,.2f}; average monthly revenue "
                       f"${f['avg_monthly_revenue']:,.2f}; DSCR proxy {dscr}; {f['nsf_count']} NSF fees; "
                       f"{f['days_negative_balance']} days negative. Recommendation: {rec}.")
            cited = [{"name": k, "value": f[k]} for k in
                     ("avg_daily_balance", "avg_monthly_revenue", "dscr_proxy", "nsf_count", "days_negative_balance")
                     if f[k] is not None]
            return self._turn([self._use("submit_underwriting_memo", {
                "recommendation": rec, "summary": summary,
                "key_risks": sorted(self._flagged) or ["none identified"], "cited_metrics": cited})])
        return self._turn([{"type": "text", "text": "Memo submitted."}], stop="end_turn")
