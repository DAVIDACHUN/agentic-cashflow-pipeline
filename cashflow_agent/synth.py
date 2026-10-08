"""SYNTHETIC small-business bank statements with golden labels.

Every statement is generated from a business archetype and carries ground
truth for evaluation:
  * a category label for every transaction,
  * the risk/fraud signals that were deliberately injected (with txn ids),
  * the recommendation the credit policy implies for that ground truth.

Business names, counterparties and amounts are invented. Nothing here is
derived from real customers or any employer/client system.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from datetime import date, timedelta
from pathlib import Path

from cashflow_agent.policy import golden_recommendation

CATEGORIES = [
    "payroll", "rent", "loan_repayment", "mca_remittance", "nsf_fee", "card_revenue",
    "customer_payment", "transfer", "utilities", "tax_payment", "supplier_payment", "insurance",
]

STEMS = {
    "payroll": ["GUSTOPAY PAYROLL", "ADPW WAGE PAY", "PAYCHEQ PAYROLL", "QBOOKS PAYROLL", "ONPAY PAYROLL"],
    "rent": ["MAPLEWOOD PROPERTY MGMT", "CROSSPOINT PLAZA LEASE", "RENT PAYMENT", "OAKRIDGE PROPERTIES"],
    "loan_repayment": ["SBA LOAN PMT", "BLUEVINE LOAN PAYMT", "TERM LOAN PAYMENT", "EQUIP FINANCE PMT"],
    "mca_remittance": ["LIBERTY FUNDING DAILY", "YELLOWSTONE CAP ACH", "PEARL CAP RECV", "RAPID ADVANCE DLY",
                       "KALAMATA CAP"],
    "card_revenue": ["SQUARE INC DEP", "STRIPE TRANSFER", "TOASTPOS DEP", "CLOVER SETTLEMENT", "SHOPIFY PAYOUT"],
    "customer_payment": ["ACH CREDIT INV", "BILL.COM RECEIVABLE", "CUSTOMER PMT INV", "MOBILE DEPOSIT"],
    "transfer": ["ONLINE TRANSFER TO SAV", "ONLINE TRANSFER FROM SAV", "ZELLE TO OWNER", "XFER FROM ACCT"],
    "utilities": ["CON ED ELECTRIC", "VERIZON BUSINESS", "NATL GRID GAS", "WASTE MGMT", "COMCAST BUSINESS"],
    "tax_payment": ["IRS USATAXPYMT", "EFTPS TAX PMT", "NYS DTF SALES TAX"],
    "supplier_payment": ["SYSCO FOODS", "RESTAURANT DEPOT", "ULINE SUPPLIES", "GRAINGER", "BILL.COM PAYABLE"],
    "insurance": ["HISCOX INS PREM", "NEXT INSURANCE", "HARTFORD INS PMT", "WORKERS COMP PREM"],
}
GENERIC = ["ACH DEBIT WEB PMT", "ELECTRONIC PAYMENT", "PREAUTHORIZED ACH", "ONLINE PAYMENT THANK YOU"]
NAMES = ["Harbor Lane Bakery", "Tidewater Auto Repair", "Bluebird Dental Studio", "Northgate Fitness",
         "Copperline Electric", "Juniper Floral Co", "Ironside Fabrication", "Saffron Table Bistro",
         "Kestrel Logistics", "Meadowbrook Pet Care", "Pinecrest Plumbing", "Lantern Bookshop",
         "Riverbend Landscaping", "Silverleaf Salon", "Granite State Movers", "Orchard Hill Market",
         "Beacon Print Shop", "Cedar Ridge Cafe", "Summit HVAC Services", "Wildflower Yoga"]

ARCHETYPES = {
    #            daily rev,  b2b?,  margin, has_loan, mca_n, start bal
    "healthy":   dict(rev=(1800, 6000), b2b=0.3, margin=(0.12, 0.25), loan=0.5, mca=0, bal=(25000, 90000)),
    "thin":      dict(rev=(1200, 4000), b2b=0.3, margin=(0.02, 0.07), loan=0.7, mca=0, bal=(6000, 20000)),
    "stressed":  dict(rev=(900, 3000), b2b=0.2, margin=(-0.06, 0.0), loan=0.6, mca=2, bal=(1500, 6000)),
    "fraud":     dict(rev=(1200, 4500), b2b=0.3, margin=(0.04, 0.15), loan=0.4, mca=0, bal=(5000, 30000)),
}
FRAUD_TYPES = ["duplicate_deposit", "balance_tampering", "structuring"]


def _desc(rng: random.Random, cat: str, stem: str | None = None) -> str:
    if cat != "nsf_fee" and rng.random() < 0.06:
        return rng.choice(GENERIC)
    stem = stem or rng.choice(STEMS[cat])
    ref = rng.choice(["", f" {rng.randint(100000, 9999999)}", f" ID:{rng.randint(10000, 99999)}"])
    return (stem + ref)[:40]


class _Book:
    def __init__(self, rng: random.Random, ctr):
        self.rng = rng
        self.ctr = ctr
        self.rows: list[dict] = []

    def add(self, d: date, cat: str, amount: float, desc: str | None = None, stem: str | None = None) -> dict:
        row = {"date": d.isoformat(), "description": desc or _desc(self.rng, cat, stem),
               "amount": round(amount, 2), "label": cat, "_ref": next(self.ctr)}
        self.rows.append(row)
        return row


def generate_statement(rng: random.Random, idx: int, archetype: str, start: date, days: int = 90) -> dict:
    a = ARCHETYPES[archetype]
    ctr = itertools.count()
    book = _Book(rng, ctr)
    rev = rng.uniform(*a["rev"])
    b2b = rng.random() < a["b2b"]
    margin = rng.uniform(*a["margin"])
    opex_daily = rev * (1 - margin)
    has_loan = rng.random() < a["loan"]
    loan_pmt = round(rev * rng.uniform(0.8, 2.0), 2)
    mca_funders = rng.sample(STEMS["mca_remittance"], a["mca"]) if a["mca"] else []
    card_stem = rng.choice(STEMS["card_revenue"])
    payroll_stem = rng.choice(STEMS["payroll"])
    rent_stem = rng.choice(STEMS["rent"])
    payroll_amt = opex_daily * 14 * 0.38
    rent_amt = round(opex_daily * 30 * 0.12, 2)

    for t in range(days):
        d = start + timedelta(days=t)
        wd = d.weekday()
        if not b2b and wd < 6:
            book.add(d, "card_revenue", rng.lognormvariate(0, 0.35) * rev * 7 / 6, stem=card_stem)
        if b2b and wd < 5 and rng.random() < 0.35:
            book.add(d, "customer_payment", rng.lognormvariate(0, 0.5) * rev * 7 / 1.75 * 0.85)
        if b2b and wd < 5 and rng.random() < 0.3:
            book.add(d, "card_revenue", rng.lognormvariate(0, 0.4) * rev * 0.4, stem=card_stem)
        if wd == 4 and (t // 7) % 2 == 0:
            book.add(d, "payroll", -payroll_amt * rng.uniform(0.95, 1.05), stem=payroll_stem)
        if d.day == 1:
            book.add(d, "rent", -rent_amt, stem=rent_stem)
            book.add(d, "insurance", -round(opex_daily * 30 * 0.02, 2))
        if d.day in (5, 18):
            book.add(d, "utilities", -opex_daily * rng.uniform(0.3, 0.6))
        if d.day == 15:
            book.add(d, "tax_payment", -opex_daily * rng.uniform(0.8, 1.6))
            if has_loan:
                book.add(d, "loan_repayment", -loan_pmt, stem="SBA LOAN PMT")
        if wd in (1, 3):
            book.add(d, "supplier_payment", -opex_daily * 3.5 * rng.uniform(0.7, 1.3) * 0.43)
        if wd < 5:
            for f in mca_funders:
                book.add(d, "mca_remittance", -round(rev * 0.09, 2), stem=f)
        if rng.random() < 0.04:
            sign = rng.choice([-1, 1])
            stem = STEMS["transfer"][0 if sign < 0 else 1]
            book.add(d, "transfer", sign * rng.uniform(500, 4000), stem=stem)

    signals: list[dict] = []
    # --- benign confounder: big customer wire then big supplier payment (looks like pass-through)
    if b2b and rng.random() < 0.5:
        t = rng.randint(10, days - 10)
        big = rev * rng.uniform(8, 14)
        d = start + timedelta(days=t)
        book.add(d, "customer_payment", big, desc=f"WIRE IN CONTRACT DRAW {rng.randint(1000, 9999)}")
        book.add(d + timedelta(days=1), "supplier_payment", -big * rng.uniform(0.85, 0.95),
                 desc=f"WIRE OUT MATERIALS VENDOR {rng.randint(1000, 9999)}")

    fraud_kind = None
    if archetype == "fraud":
        fraud_kind = FRAUD_TYPES[idx % len(FRAUD_TYPES)]
    if archetype == "stressed" and rng.random() < 0.5:
        # pass-through of borrowed money: large inbound transfer, quickly sent out to owner
        t = rng.randint(15, days - 10)
        big = round(rev * rng.uniform(10, 16), 2)
        d = start + timedelta(days=t)
        r1 = book.add(d, "transfer", big, desc=f"WIRE IN {rng.randint(100000, 999999)}")
        r2 = book.add(d + timedelta(days=rng.randint(0, 2)), "transfer", -big * rng.uniform(0.9, 1.0),
                      desc="ZELLE TO OWNER")
        signals.append({"type": "pass_through_funds", "txn_refs": [r1["_ref"], r2["_ref"]]})

    if fraud_kind == "structuring":
        refs = []
        for _ in range(rng.randint(3, 5)):
            d = start + timedelta(days=rng.randint(5, days - 5))
            refs.append(book.add(d, "customer_payment", rng.uniform(9000, 9950), desc="CASH DEPOSIT BRANCH")["_ref"])
        signals.append({"type": "structuring", "txn_refs": refs})

    # sort, then simulate the running balance with natural NSF fees
    book.rows.sort(key=lambda r: (r["date"], r["amount"] < 0))
    bal = round(rng.uniform(*a["bal"]), 2)
    opening = bal
    out: list[dict] = []
    nsf_today: set[str] = set()
    for r in book.rows:
        bal = round(bal + r["amount"], 2)
        out.append(dict(r))
        if r["amount"] < 0 and bal < 0 and r["date"] not in nsf_today:
            nsf_today.add(r["date"])
            fee = {"date": r["date"], "description": rng.choice(["NSF FEE", "OVERDRAFT ITEM FEE",
                                                                "RETURNED ITEM FEE"]),
                   "amount": -35.0, "label": "nsf_fee", "_ref": next(ctr)}
            bal = round(bal - 35.0, 2)
            out.append(fee)

    if fraud_kind == "duplicate_deposit":
        credits = [i for i, r in enumerate(out) if r["label"] == "card_revenue" or r["label"] == "customer_payment"]
        picks = sorted(rng.sample(credits, k=min(3, len(credits))))
        refs = []
        for k, i in enumerate(picks):  # insert duplicates right after originals
            dup = {**out[i + k], "_ref": next(ctr)}
            out.insert(i + k + 1, dup)
            refs.append(dup["_ref"])
        signals.append({"type": "duplicate_deposit", "txn_refs": refs})

    # running balances (recomputed so injected duplicates flow through honestly)
    bal = opening
    for r in out:
        bal = round(bal + r["amount"], 2)
        r["running_balance"] = bal

    if fraud_kind == "balance_tampering":
        i0 = rng.randint(len(out) // 3, 2 * len(out) // 3)
        bump = round(rng.uniform(8000, 25000), 2)
        for r in out[i0:]:
            r["running_balance"] = round(r["running_balance"] + bump, 2)
        signals.append({"type": "balance_tampering", "txn_refs": [out[i0]["_ref"]]})

    nsf_refs = [r["_ref"] for r in out if r["label"] == "nsf_fee"]
    if len(nsf_refs) >= 3:
        signals.append({"type": "nsf_cluster", "txn_refs": nsf_refs})
    if len(mca_funders) >= 2:
        signals.append({"type": "mca_stacking",
                        "txn_refs": [r["_ref"] for r in out if r["label"] == "mca_remittance"][:10]})

    # assign ids, strip private refs
    ref2id = {}
    txns, labels = [], {}
    for n, r in enumerate(out):
        tid = f"T{n + 1:04d}"
        ref2id[r["_ref"]] = tid
        txns.append({"txn_id": tid, "date": r["date"], "description": r["description"],
                     "amount": r["amount"], "running_balance": r["running_balance"]})
        labels[tid] = r["label"]
    for s in signals:
        s["txn_ids"] = [ref2id[x] for x in s.pop("txn_refs")]

    statement = {
        "statement_id": f"STMT-{idx:03d}",
        "business_name": f"{rng.choice(NAMES)} LLC (synthetic)",
        "period_start": start.isoformat(),
        "period_end": (start + timedelta(days=days - 1)).isoformat(),
        "opening_balance": opening,
        "transactions": txns,
    }
    golden = {"statement_id": statement["statement_id"], "archetype": archetype,
              "labels": labels, "signals": signals}
    golden["recommendation"] = golden_recommendation(statement, golden)
    return {"statement": statement, "golden": golden}


def generate_set(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    mix = (["healthy"] * 7 + ["thin"] * 5 + ["stressed"] * 4 + ["fraud"] * 4)
    out = []
    for i in range(n):
        start = date(2026, 1, 1) + timedelta(days=rng.randint(0, 150))
        out.append(generate_statement(rng, i + 1, mix[i % len(mix)], start))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SYNTHETIC statements + golden labels")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data/golden")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "statements").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    for item in generate_set(args.n, args.seed):
        sid = item["statement"]["statement_id"]
        (out / "statements" / f"{sid}.json").write_text(json.dumps(item["statement"], indent=1))
        (out / "labels" / f"{sid}.json").write_text(json.dumps(item["golden"], indent=1))
    print(f"wrote {args.n} synthetic statements to {out}")


if __name__ == "__main__":
    main()
