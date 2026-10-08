"""Tool definitions (Claude tool-use JSON schemas) and their executors.

Each agent run owns a RunState. Tools read the statement, record the agent's
classifications / flags / memo, and run deterministic code (features,
detectors). Tool inputs are validated here regardless of backend.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from cashflow_agent import detectors
from cashflow_agent.features import compute_features
from cashflow_agent.policy import ALL_SIGNALS
from cashflow_agent.synth import CATEGORIES

PAGE_SIZE = 80


def _strict(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {"type": "object", "properties": properties,
                         "required": required if required is not None else list(properties),
                         "additionalProperties": False},
    }


TOOLS = [
    _strict("get_statement_overview",
            "Return business name, statement period, opening balance and the number of transactions. Call first.",
            {}),
    _strict("get_transactions",
            f"Return up to {PAGE_SIZE} transactions starting at `offset` (0-based). Each has txn_id, date, "
            "description, signed amount (+ credit / - debit) and the running balance printed on the statement.",
            {"offset": {"type": "integer"}}),
    _strict("classify_transactions",
            "Record a category and a confidence in [0,1] for each transaction. Confidence must reflect real "
            "uncertainty: generic descriptors (e.g. 'ELECTRONIC PAYMENT') deserve low confidence. Every "
            "transaction must be classified before features or anomaly checks are run.",
            {"items": {"type": "array", "items": {
                "type": "object",
                "properties": {"txn_id": {"type": "string"},
                               "category": {"type": "string", "enum": CATEGORIES},
                               "confidence": {"type": "number"}},
                "required": ["txn_id", "category", "confidence"],
                "additionalProperties": False}}}),
    _strict("run_anomaly_checks",
            "Run deterministic fraud/risk detectors over the statement and your classifications. Returns "
            "CANDIDATE signals with evidence; they can be false positives. Confirm real ones with flag_risk.",
            {}),
    _strict("flag_risk",
            "Confirm a risk or fraud signal. Use only after reviewing the evidence.",
            {"signal_type": {"type": "string", "enum": ALL_SIGNALS},
             "txn_ids": {"type": "array", "items": {"type": "string"}},
             "severity": {"type": "string", "enum": ["low", "medium", "high"]},
             "evidence": {"type": "string"}}),
    _strict("compute_cashflow_features",
            "Compute cash-flow features from the statement and your classifications: average daily balance, "
            "days negative, NSF count, monthly revenue/opex/debt service, weekly revenue volatility, "
            "debt-service-coverage proxy, MCA debit count, transfer share of credits.",
            {}),
    _strict("submit_underwriting_memo",
            "Submit the final memo. Call exactly once, last. Quote metric values exactly as returned by "
            "compute_cashflow_features.",
            {"recommendation": {"type": "string", "enum": ["approve", "refer", "decline"]},
             "summary": {"type": "string"},
             "key_risks": {"type": "array", "items": {"type": "string"}},
             "cited_metrics": {"type": "array", "items": {
                 "type": "object",
                 "properties": {"name": {"type": "string"}, "value": {"type": "number"}},
                 "required": ["name", "value"], "additionalProperties": False}}}),
]
TOOL_NAMES = {t["name"] for t in TOOLS}


class ToolError(Exception):
    pass


@dataclass
class RunState:
    statement: dict
    classifications: dict[str, dict] = field(default_factory=dict)
    flags: list[dict] = field(default_factory=list)
    features: dict | None = None
    candidates: list[dict] | None = None
    memo: dict | None = None
    tool_calls: list[str] = field(default_factory=list)

    @property
    def labels(self) -> dict[str, str]:
        return {k: v["category"] for k, v in self.classifications.items()}

    def _txn_ids(self) -> set[str]:
        return {t["txn_id"] for t in self.statement["transactions"]}

    def _require_complete(self) -> None:
        missing = self._txn_ids() - set(self.classifications)
        if missing:
            raise ToolError(f"{len(missing)} transactions not yet classified, e.g. {sorted(missing)[:5]}")

    def execute(self, name: str, args: dict) -> dict:
        self.tool_calls.append(name)
        if name not in TOOL_NAMES:
            raise ToolError(f"unknown tool {name}")
        s = self.statement
        if name == "get_statement_overview":
            return {k: s[k] for k in ("statement_id", "business_name", "period_start", "period_end",
                                      "opening_balance")} | {"n_transactions": len(s["transactions"]),
                                                             "page_size": PAGE_SIZE}
        if name == "get_transactions":
            off = int(args["offset"])
            page = s["transactions"][off: off + PAGE_SIZE]
            nxt = off + PAGE_SIZE if off + PAGE_SIZE < len(s["transactions"]) else None
            return {"transactions": page, "next_offset": nxt}
        if name == "classify_transactions":
            valid, bad = self._txn_ids(), []
            for it in args["items"]:
                if it["txn_id"] not in valid or it["category"] not in CATEGORIES:
                    bad.append(it.get("txn_id"))
                    continue
                conf = min(max(float(it["confidence"]), 0.0), 1.0)
                self.classifications[it["txn_id"]] = {"category": it["category"], "confidence": conf}
            return {"recorded": len(args["items"]) - len(bad), "rejected": bad,
                    "remaining_unclassified": len(valid - set(self.classifications))}
        if name == "run_anomaly_checks":
            self._require_complete()
            self.candidates = detectors.run_all(s, self.labels)
            return {"candidates": self.candidates}
        if name == "flag_risk":
            unknown = [x for x in args["txn_ids"] if x not in self._txn_ids()]
            if unknown:
                raise ToolError(f"unknown txn_ids {unknown[:5]}")
            self.flags.append(dict(args))
            return {"flag_recorded": len(self.flags)}
        if name == "compute_cashflow_features":
            self._require_complete()
            self.features = compute_features(s, self.labels)
            return self.features
        if name == "submit_underwriting_memo":
            if self.features is None:
                raise ToolError("call compute_cashflow_features before submitting the memo")
            self.memo = dict(args)
            return {"status": "submitted"}
        raise ToolError(name)


def tool_result_block(tool_use_id: str, payload: dict | str, is_error: bool = False) -> dict:
    return {"type": "tool_result", "tool_use_id": tool_use_id,
            "content": payload if isinstance(payload, str) else json.dumps(payload, default=str),
            "is_error": is_error}
