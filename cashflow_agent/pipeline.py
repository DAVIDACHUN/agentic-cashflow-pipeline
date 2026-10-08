"""Run the agent over a folder of statements, persist outputs, fill the review queue.

    python -m cashflow_agent.pipeline --backend mock   --out runs/mock
    python -m cashflow_agent.pipeline --backend claude --model claude-haiku-5-5 --out runs/haiku
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cashflow_agent import review_queue
from cashflow_agent.agent import run_agent
from cashflow_agent.llm import ClaudeBackend, MockBackend


def make_backend(kind: str, model: str, effort: str):
    return MockBackend() if kind == "mock" else ClaudeBackend(model=model, effort=effort)


def run(statements_dir: Path, out: Path, backend_kind: str, model: str, effort: str, limit: int | None = None) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(exist_ok=True)
    db = out / "review_queue.db"
    db.unlink(missing_ok=True)
    con = review_queue.connect(str(db))
    files = sorted(statements_dir.glob("*.json"))[:limit]
    summaries = []
    for f in files:
        stmt = json.loads(f.read_text())
        backend = make_backend(backend_kind, model, effort)  # fresh state per statement
        t0 = time.time()
        res = run_agent(stmt, backend)
        queued = review_queue.route_result(con, res)
        rec = {
            "statement_id": res.statement_id, "backend": backend.name, "status": res.status,
            "turns": res.turns, "tool_errors": res.tool_errors, "seconds": round(time.time() - t0, 2),
            "usage": res.usage, "tool_calls": res.state.tool_calls,
            "classifications": res.state.classifications, "flags": res.state.flags,
            "features": res.state.features, "memo": res.state.memo, "candidates": res.state.candidates,
            "queued": queued,
        }
        (out / "results" / f"{res.statement_id}.json").write_text(json.dumps(rec, indent=1, default=str))
        summaries.append(rec)
        memo = (res.state.memo or {}).get("recommendation", "-")
        print(f"{res.statement_id} {res.status:<10} turns={res.turns:<3} memo={memo:<8} queued={queued}")
    return summaries


def rebuild_queue(statements_dir: Path, out: Path) -> None:
    """Re-apply the current routing rules to a saved run, with no LLM calls. Detectors are deterministic,
    so candidates missing from older result files are recomputed from the agent's saved classifications."""
    from cashflow_agent import detectors

    db = out / "review_queue.db"
    db.unlink(missing_ok=True)
    con = review_queue.connect(str(db))
    for f in sorted((out / "results").glob("*.json")):
        rec = json.loads(f.read_text())
        stmt = json.loads((statements_dir / f"{rec['statement_id']}.json").read_text())
        if rec.get("candidates") is None:
            labels = {k: v["category"] for k, v in rec["classifications"].items()}
            rec["candidates"] = detectors.run_all(stmt, labels)
        rec["queued"] = review_queue.route_record(con, rec, stmt)
        f.write_text(json.dumps(rec, indent=1, default=str))
        print(rec["statement_id"], rec["queued"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--statements", default="data/golden/statements")
    ap.add_argument("--out", default="runs/mock")
    ap.add_argument("--backend", choices=["mock", "claude"], default="mock")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--rebuild-queue", action="store_true", help="re-route a saved run in --out; no LLM calls")
    a = ap.parse_args()
    if a.rebuild_queue:
        return rebuild_queue(Path(a.statements), Path(a.out))
    run(Path(a.statements), Path(a.out), a.backend, a.model, a.effort, a.limit)


if __name__ == "__main__":
    main()
