"""Deterministic small-business cash-flow features.

Computed in Python from the statement + per-transaction categories. The LLM
never does this arithmetic; it calls a tool that runs this code.
"""
from __future__ import annotations

from datetime import date, timedelta
from statistics import mean, pstdev

REVENUE = {"card_revenue", "customer_payment"}
OPEX = {"payroll", "rent", "utilities", "supplier_payment", "insurance", "tax_payment"}
DEBT_SERVICE = {"loan_repayment", "mca_remittance"}


def _daily_balances(statement: dict) -> list[float]:
    start = date.fromisoformat(statement["period_start"])
    end = date.fromisoformat(statement["period_end"])
    eod: dict[str, float] = {}
    for t in statement["transactions"]:  # statements are date-ordered; last row of the day wins
        eod[t["date"]] = t["running_balance"]
    bal, out, d = statement["opening_balance"], [], start
    while d <= end:
        bal = eod.get(d.isoformat(), bal)
        out.append(bal)
        d += timedelta(days=1)
    return out


def compute_features(statement: dict, labels: dict[str, str]) -> dict:
    txns = statement["transactions"]
    start = date.fromisoformat(statement["period_start"])
    days = (date.fromisoformat(statement["period_end"]) - start).days + 1
    per_month = 30.0 / days

    def total(cats: set[str], sign: int) -> float:
        return sum(abs(t["amount"]) for t in txns
                   if labels.get(t["txn_id"]) in cats and (t["amount"] > 0) == (sign > 0))

    revenue = total(REVENUE, +1)
    opex = total(OPEX, -1)
    debt = total(DEBT_SERVICE, -1)
    credits = sum(t["amount"] for t in txns if t["amount"] > 0)
    transfers_in = total({"transfer"}, +1)

    weeks = [0.0] * ((days + 6) // 7)
    for t in txns:
        if labels.get(t["txn_id"]) in REVENUE and t["amount"] > 0:
            weeks[(date.fromisoformat(t["date"]) - start).days // 7] += t["amount"]
    full_weeks = weeks[: days // 7]
    cv = pstdev(full_weeks) / mean(full_weeks) if full_weeks and mean(full_weeks) > 0 else None

    bals = _daily_balances(statement)
    noi_m = (revenue - opex) * per_month
    debt_m = debt * per_month
    return {
        "period_days": days,
        "avg_daily_balance": round(mean(bals), 2),
        "min_daily_balance": round(min(bals), 2),
        "days_negative_balance": sum(b < 0 for b in bals),
        "nsf_count": sum(labels.get(t["txn_id"]) == "nsf_fee" for t in txns),
        "avg_monthly_revenue": round(revenue * per_month, 2),
        "revenue_volatility_weekly_cv": round(cv, 4) if cv is not None else None,
        "avg_monthly_opex": round(opex * per_month, 2),
        "avg_monthly_debt_service": round(debt_m, 2),
        "dscr_proxy": round(noi_m / debt_m, 3) if debt_m > 0 else None,
        "mca_debit_count": sum(labels.get(t["txn_id"]) == "mca_remittance" for t in txns),
        "transfer_share_of_credits": round(transfers_in / credits, 4) if credits else 0.0,
    }
