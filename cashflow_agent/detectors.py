"""Deterministic anomaly / fraud *candidate* detectors.

Detectors are deliberately recall-oriented: they surface candidates with
evidence, and the agent decides which to confirm as flags. Some detectors
depend on the agent's classifications (e.g. pass-through needs to know an
outflow was a transfer, not a supplier payment), so classification quality
propagates into detection quality - the eval measures that.
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from statistics import median

STRUCTURING_BAND = (9000.0, 10000.0)


def _norm(desc: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"(ID:)?\d+", "", desc.upper())).strip()


def duplicate_deposits(statement: dict, labels: dict) -> list[dict]:
    seen: dict[tuple, str] = {}
    out = []
    for t in statement["transactions"]:
        if t["amount"] <= 0:
            continue
        key = (t["date"], t["description"], t["amount"])
        if key in seen:
            out.append({"type": "duplicate_deposit", "txn_ids": [t["txn_id"]],
                        "evidence": f"identical credit to {seen[key]} (same date, descriptor, amount {t['amount']:.2f})"})
        else:
            seen[key] = t["txn_id"]
    return _merge(out)


def balance_tampering(statement: dict, labels: dict) -> list[dict]:
    bal = statement["opening_balance"]
    for t in statement["transactions"]:
        bal = round(bal + t["amount"], 2)
        if abs(bal - t["running_balance"]) > 0.01:
            return [{"type": "balance_tampering", "txn_ids": [t["txn_id"]],
                     "evidence": f"running balance {t['running_balance']:.2f} != opening + cumulative amounts "
                                 f"{bal:.2f} (gap {t['running_balance'] - bal:+.2f}) from this row on"}]
    return []


def structuring(statement: dict, labels: dict) -> list[dict]:
    lo, hi = STRUCTURING_BAND
    hits = [t for t in statement["transactions"] if lo <= t["amount"] < hi]
    hits.sort(key=lambda t: t["date"])
    for i in range(len(hits)):
        window = [h for h in hits[i:] if (date.fromisoformat(h["date"]) - date.fromisoformat(hits[i]["date"])).days <= 30]
        if len(window) >= 3:
            return [{"type": "structuring", "txn_ids": [h["txn_id"] for h in hits],
                     "evidence": f"{len(window)} credits in ${lo:,.0f}-${hi:,.0f} within 30 days: "
                                 + "; ".join(f"{h['txn_id']} {h['description']} {h['amount']:.2f}" for h in window)}]
    return []


def pass_through(statement: dict, labels: dict) -> list[dict]:
    txns = statement["transactions"]
    rev = [t["amount"] for t in txns if labels.get(t["txn_id"]) in ("card_revenue", "customer_payment")]
    typical = median(rev) if rev else 1000.0
    out = []
    for i, c in enumerate(txns):
        if c["amount"] < 5 * typical:
            continue
        cd = date.fromisoformat(c["date"])
        for d in txns[i + 1:]:
            if (date.fromisoformat(d["date"]) - cd).days > 3:
                break
            if (d["amount"] < 0 and labels.get(d["txn_id"]) == "transfer"
                    and abs(abs(d["amount"]) - c["amount"]) <= 0.15 * c["amount"]):
                out.append({"type": "pass_through_funds", "txn_ids": [c["txn_id"], d["txn_id"]],
                            "evidence": f"credit {c['amount']:.2f} ({c['description']}) followed by transfer out "
                                        f"{d['amount']:.2f} ({d['description']}) within 3 days"})
                break
    return _merge(out)


def nsf_cluster(statement: dict, labels: dict) -> list[dict]:
    ids = [t["txn_id"] for t in statement["transactions"] if labels.get(t["txn_id"]) == "nsf_fee"]
    if len(ids) >= 3:
        return [{"type": "nsf_cluster", "txn_ids": ids, "evidence": f"{len(ids)} NSF/overdraft fees in period"}]
    return []


def mca_stacking(statement: dict, labels: dict) -> list[dict]:
    by_funder: dict[str, list[str]] = defaultdict(list)
    for t in statement["transactions"]:
        if labels.get(t["txn_id"]) == "mca_remittance":
            by_funder[_norm(t["description"])].append(t["txn_id"])
    if len(by_funder) >= 2:
        return [{"type": "mca_stacking", "txn_ids": [v[0] for v in by_funder.values()],
                 "evidence": f"{len(by_funder)} distinct MCA-style daily debit counterparties: "
                             + ", ".join(f"{k} (x{len(v)})" for k, v in by_funder.items())}]
    return []


DETECTORS = [duplicate_deposits, balance_tampering, structuring, pass_through, nsf_cluster, mca_stacking]


def run_all(statement: dict, labels: dict) -> list[dict]:
    out = []
    for det in DETECTORS:
        out.extend(det(statement, labels))
    for i, c in enumerate(out):
        c["candidate_id"] = f"C{i + 1}"
    return out


def _merge(cands: list[dict]) -> list[dict]:
    if not cands:
        return []
    return [{"type": cands[0]["type"], "txn_ids": [x for c in cands for x in c["txn_ids"]],
             "evidence": " | ".join(c["evidence"] for c in cands)}]
