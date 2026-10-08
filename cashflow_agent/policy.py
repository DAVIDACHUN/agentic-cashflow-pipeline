"""Illustrative credit policy used both to label the golden set and to brief the
agent. It is a toy policy for demonstration, not any lender's actual policy."""
from __future__ import annotations

FRAUD_SIGNALS = {"duplicate_deposit", "balance_tampering", "structuring"}
REFER_SIGNALS = {"mca_stacking", "nsf_cluster", "pass_through_funds"}
ALL_SIGNALS = sorted(FRAUD_SIGNALS | REFER_SIGNALS)

MIN_DSCR = 1.25
MAX_DAYS_NEGATIVE = 5

POLICY_TEXT = f"""Credit policy (illustrative):
1. DECLINE if any fraud signal is confirmed: {', '.join(sorted(FRAUD_SIGNALS))}.
2. Otherwise REFER to a credit officer if any of: a confirmed risk signal in
   {', '.join(sorted(REFER_SIGNALS))}; debt-service-coverage proxy below {MIN_DSCR}
   (when the business has debt service); more than {MAX_DAYS_NEGATIVE} days with a negative balance.
3. Otherwise APPROVE."""


def recommend(signal_types: set[str], features: dict) -> str:
    if signal_types & FRAUD_SIGNALS:
        return "decline"
    dscr = features.get("dscr_proxy")
    if (signal_types & REFER_SIGNALS
            or (dscr is not None and dscr < MIN_DSCR)
            or features.get("days_negative_balance", 0) > MAX_DAYS_NEGATIVE):
        return "refer"
    return "approve"


def golden_recommendation(statement: dict, golden: dict) -> str:
    from cashflow_agent.features import compute_features

    feats = compute_features(statement, golden["labels"])
    golden["features"] = feats
    return recommend({s["type"] for s in golden["signals"]}, feats)
