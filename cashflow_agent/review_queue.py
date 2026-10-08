"""Human-in-the-loop review queue (SQLite).

Routing rules (see `route_result`):
  * classification with confidence < CONF_THRESHOLD   -> review the label
  * every confirmed risk flag of severity medium/high  -> review the flag
  * memo whose recommendation is refer/decline, or that
    coexists with a high-severity flag                 -> credit-officer sign-off
  * agent run that refused / truncated / hit turn cap  -> review whole statement

CLI:
  python -m cashflow_agent.review_queue --db runs/mock/review_queue.db list
  python -m cashflow_agent.review_queue --db ... resolve 12 --decision corrected --note "is rent" \
      --correction '{"category": "rent"}'
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone

CONF_THRESHOLD = 0.80

SCHEMA = """
CREATE TABLE IF NOT EXISTS review_items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    statement_id  TEXT NOT NULL,
    item_type     TEXT NOT NULL CHECK (item_type IN ('classification','flag','memo','statement')),
    ref           TEXT,
    payload       TEXT NOT NULL,
    reason        TEXT NOT NULL,
    priority      INTEGER NOT NULL,
    status        TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','approved','corrected','rejected')),
    reviewer      TEXT,
    resolution    TEXT,
    created_at    TEXT NOT NULL,
    resolved_at   TEXT
);
CREATE INDEX IF NOT EXISTS ix_open ON review_items(status, priority);
"""


def connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def enqueue(con, statement_id, item_type, ref, payload, reason, priority) -> None:
    con.execute("INSERT INTO review_items (statement_id,item_type,ref,payload,reason,priority,created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (statement_id, item_type, ref, json.dumps(payload, default=str), reason, priority, _now()))


def route_result(con, result) -> dict:
    """Push everything that needs a human into the queue. Returns counts."""
    sid, st = result.statement_id, result.state
    counts = {"classification": 0, "flag": 0, "memo": 0, "statement": 0}
    if result.status != "completed":
        enqueue(con, sid, "statement", None, {"status": result.status}, f"agent run {result.status}", 0)
        counts["statement"] += 1
    txn = {t["txn_id"]: t for t in st.statement["transactions"]}
    for tid, c in st.classifications.items():
        if c["confidence"] < CONF_THRESHOLD:
            enqueue(con, sid, "classification", tid, {**c, "txn": txn[tid]},
                    f"confidence {c['confidence']:.2f} < {CONF_THRESHOLD}", 3)
            counts["classification"] += 1
    high = False
    for f in st.flags:
        if f["severity"] in ("medium", "high"):
            enqueue(con, sid, "flag", f["signal_type"], f, f"{f['severity']} severity {f['signal_type']}",
                    1 if f["severity"] == "high" else 2)
            counts["flag"] += 1
        high |= f["severity"] == "high"
    if st.memo and (st.memo["recommendation"] != "approve" or high):
        reason = "adverse recommendation needs sign-off" if st.memo["recommendation"] != "approve" \
            else "approve recommended despite high-severity flag"
        enqueue(con, sid, "memo", st.memo["recommendation"], st.memo, reason, 0 if high else 1)
        counts["memo"] += 1
    con.commit()
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--status", default="open")
    ls.add_argument("--limit", type=int, default=25)
    rs = sub.add_parser("resolve")
    rs.add_argument("item_id", type=int)
    rs.add_argument("--decision", choices=["approved", "corrected", "rejected"], required=True)
    rs.add_argument("--reviewer", default="analyst")
    rs.add_argument("--note", default="")
    rs.add_argument("--correction", default="{}", help="JSON, e.g. '{\"category\": \"rent\"}'")
    sub.add_parser("stats")
    a = ap.parse_args()
    con = connect(a.db)
    if a.cmd == "list":
        rows = con.execute("SELECT id,statement_id,item_type,ref,reason,priority FROM review_items "
                           "WHERE status=? ORDER BY priority, id LIMIT ?", (a.status, a.limit)).fetchall()
        for r in rows:
            print(f"#{r[0]:<5} P{r[5]} {r[1]}  {r[2]:<14} {str(r[3] or ''):<18} {r[4]}")
    elif a.cmd == "resolve":
        res = {"note": a.note, "correction": json.loads(a.correction)}
        n = con.execute("UPDATE review_items SET status=?, reviewer=?, resolution=?, resolved_at=? "
                        "WHERE id=? AND status='open'",
                        (a.decision, a.reviewer, json.dumps(res), _now(), a.item_id)).rowcount
        con.commit()
        print("resolved" if n else "no open item with that id")
    else:
        for r in con.execute("SELECT item_type,status,COUNT(*) FROM review_items GROUP BY 1,2 ORDER BY 1,2"):
            print(f"{r[0]:<15} {r[1]:<10} {r[2]}")


if __name__ == "__main__":
    main()
