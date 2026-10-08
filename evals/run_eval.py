"""Evaluate an agent run against the golden set.

    python -m evals.run_eval --backend mock   --judge heuristic          # offline
    python -m evals.run_eval --backend claude --judge claude --name haiku  # live (needs credentials)

Measures: classification quality (overall / auto-accepted / queued), effect of
HITL routing, risk-signal precision & recall, feature error propagated from
misclassification, recommendation accuracy incl. unsafe approvals, judge pass
rate and judge reliability on negative controls, and cost/latency.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from sklearn.metrics import f1_score

from cashflow_agent.pipeline import run as run_pipeline
from cashflow_agent.policy import ALL_SIGNALS
from cashflow_agent.review_queue import CONF_THRESHOLD
from evals.judge import ClaudeJudge, HeuristicJudge, corrupt

PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}  # $/M in, out
KEY_FEATURES = ["avg_daily_balance", "avg_monthly_revenue", "avg_monthly_opex", "avg_monthly_debt_service",
                "dscr_proxy", "nsf_count", "revenue_volatility_weekly_cv"]


def evaluate(golden_dir: Path, run_dir: Path, judge) -> dict:
    goldens = {p.stem: json.loads(p.read_text()) for p in sorted((golden_dir / "labels").glob("*.json"))}
    results = {p.stem: json.loads(p.read_text()) for p in sorted((run_dir / "results").glob("*.json"))}
    ids = [s for s in goldens if s in results]
    for s in ids:  # statement-level facts for the judge
        st = json.loads((golden_dir / "statements" / f"{s}.json").read_text())
        goldens[s]["overview"] = {
            "business_name": st["business_name"], "period_start": st["period_start"], "period_end": st["period_end"],
            "n_transactions": len(st["transactions"]), "opening_balance": st["opening_balance"],
            "closing_balance": st["transactions"][-1]["running_balance"]}

    # ---- classification
    y, yhat, auto_ok, auto_n, q_ok, q_n = [], [], 0, 0, 0, 0
    for s in ids:
        for tid, lab in goldens[s]["labels"].items():
            c = results[s]["classifications"].get(tid)
            pred = c["category"] if c else "__missing__"
            y.append(lab)
            yhat.append(pred)
            ok = pred == lab
            if c and c["confidence"] >= CONF_THRESHOLD:
                auto_n, auto_ok = auto_n + 1, auto_ok + ok
            else:
                q_n, q_ok = q_n + 1, q_ok + ok
    n = len(y)
    cls = {
        "n_transactions": n,
        "accuracy": sum(a == b for a, b in zip(y, yhat)) / n,
        "macro_f1": f1_score(y, yhat, average="macro", labels=sorted(set(y))),
        "queue_rate": q_n / n,
        "auto_accepted_accuracy": auto_ok / auto_n if auto_n else None,
        "queued_model_accuracy": q_ok / q_n if q_n else None,
        # if reviewers correct every queued item, residual errors are the confident mistakes
        "accuracy_after_hitl": (auto_ok + q_n) / n,
        "per_class_f1": dict(zip(sorted(set(y)), f1_score(y, yhat, average=None, labels=sorted(set(y))).round(3).tolist())),
    }

    # ---- signals (statement-level presence per type)
    tp, fp, fn = Counter(), Counter(), Counter()
    for s in ids:
        gt = {x["type"] for x in goldens[s]["signals"]}
        pr = {f["signal_type"] for f in results[s]["flags"]}
        for t in ALL_SIGNALS:
            tp[t] += t in gt and t in pr
            fp[t] += t in pr and t not in gt
            fn[t] += t in gt and t not in pr
    sig = {t: {"tp": tp[t], "fp": fp[t], "fn": fn[t],
               "precision": tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else None,
               "recall": tp[t] / (tp[t] + fn[t]) if tp[t] + fn[t] else None} for t in ALL_SIGNALS}
    T, P, F = sum(tp.values()), sum(fp.values()), sum(fn.values())
    sig["_overall"] = {"tp": T, "fp": P, "fn": F, "precision": T / (T + P) if T + P else None,
                       "recall": T / (T + F) if T + F else None}

    # ---- feature error vs features computed on golden labels
    feat_err = defaultdict(list)
    for s in ids:
        g, r = goldens[s]["features"], results[s]["features"] or {}
        for k in KEY_FEATURES:
            if g.get(k) is not None and r.get(k) is not None and g[k] != 0:
                feat_err[k].append(abs(r[k] - g[k]) / abs(g[k]))
    feats = {k: {"median_abs_pct_err": median(v), "max_abs_pct_err": max(v)} for k, v in feat_err.items()}

    # ---- recommendation
    conf = Counter()
    unsafe, unsafe_caught = 0, 0
    for s in ids:
        g = goldens[s]["recommendation"]
        p = (results[s]["memo"] or {}).get("recommendation", "none")
        conf[f"{g}->{p}"] += 1
        if g != "approve" and p == "approve":
            unsafe += 1
            unsafe_caught += results[s]["queued"]["memo"] > 0 or results[s]["queued"]["flag"] > 0
    rec = {"accuracy": sum(v for k, v in conf.items() if k.split("->")[0] == k.split("->")[1]) / len(ids),
           "confusion": dict(sorted(conf.items())), "unsafe_approvals": unsafe,
           "unsafe_approvals_with_any_human_touchpoint": unsafe_caught}

    # ---- judge on real memos + negative controls
    verdicts, controls, per_memo = [], defaultdict(list), {}
    for s in ids:
        memo = results[s]["memo"]
        if not memo:
            continue
        v = judge.judge(memo, goldens[s])
        verdicts.append(v)
        per_memo[s] = v.model_dump() if v is not None else {"refused": True}
        for kind, bad in corrupt(memo, goldens[s]).items():
            cv = judge.judge(bad, goldens[s])
            controls[kind].append(cv is not None and not cv.passed)
    ok = [v for v in verdicts if v is not None]
    jud = {"judge": judge.name, "n_memos": len(verdicts), "judge_refusals": len(verdicts) - len(ok),
           "pass_rate": mean(v.passed for v in ok) if ok else None,
           "mean_grounded": mean(v.grounded for v in ok) if ok else None,
           "mean_risk_coverage": mean(v.risk_coverage for v in ok) if ok else None,
           "negative_control_catch_rate": {k: mean(v) for k, v in controls.items()},
           "verdicts": per_memo}

    # ---- ops
    rs = [results[s] for s in ids]
    tok_in = sum(r["usage"].get("input_tokens", 0) + r["usage"].get("cache_read_input_tokens", 0)
                 + r["usage"].get("cache_creation_input_tokens", 0) for r in rs)
    tok_out = sum(r["usage"].get("output_tokens", 0) for r in rs)
    backend = rs[0]["backend"] if rs else "?"
    pin, pout = PRICES.get(backend.split(":")[-1], (0, 0))
    ops = {"backend": backend, "statements": len(ids),
           "status": dict(Counter(r["status"] for r in rs)),
           "mean_turns": mean(r["turns"] for r in rs), "tool_errors": sum(r["tool_errors"] for r in rs),
           "mean_seconds_per_statement": mean(r["seconds"] for r in rs),
           "input_tokens": tok_in, "output_tokens": tok_out,
           "approx_cost_usd_upper_bound": round(tok_in / 1e6 * pin + tok_out / 1e6 * pout, 4),
           "review_items": dict(sum((Counter(r["queued"]) for r in rs), Counter()))}

    ju = getattr(judge, "usage", None)
    if ju:
        jin, jout = PRICES.get(judge.name.split(":")[-1], (0, 0))
        ops["judge_usage"] = ju
        ops["judge_cost_usd_upper_bound"] = round(ju["input_tokens"] / 1e6 * jin + ju["output_tokens"] / 1e6 * jout, 4)

    return {"classification": cls, "signals": sig, "features": feats, "recommendation": rec,
            "judge": jud, "ops": ops}


def to_markdown(r: dict) -> str:
    c, s, rec, j, o = r["classification"], r["signals"], r["recommendation"], r["judge"], r["ops"]
    pct = lambda x: "n/a" if x is None else f"{x:.1%}"  # noqa: E731
    L = [f"#### Backend `{o['backend']}` | judge `{j['judge']}` | {o['statements']} statements, "
         f"{c['n_transactions']} transactions", "",
         "| Metric | Value |", "|---|---|",
         f"| Classification accuracy | {pct(c['accuracy'])} |",
         f"| Classification macro-F1 | {c['macro_f1']:.3f} |",
         f"| Routed to human review (conf < {CONF_THRESHOLD}) | {pct(c['queue_rate'])} |",
         f"| Accuracy on auto-accepted | {pct(c['auto_accepted_accuracy'])} |",
         f"| Model accuracy on queued items | {pct(c['queued_model_accuracy'])} |",
         f"| Accuracy after HITL (reviewers fix queued) | {pct(c['accuracy_after_hitl'])} |",
         f"| Risk-signal precision / recall (statement level) | {pct(s['_overall']['precision'])} / {pct(s['_overall']['recall'])} |",
         f"| Recommendation accuracy vs policy | {pct(rec['accuracy'])} |",
         f"| Unsafe approvals (policy said refer/decline) | {rec['unsafe_approvals']} |",
         f"| Judge pass rate | {pct(j['pass_rate'])} |",
         f"| Mean turns / statement | {o['mean_turns']:.1f} |",
         f"| Tool-call errors | {o['tool_errors']} |",
         f"| Agent tokens in / out | {o['input_tokens']:,} / {o['output_tokens']:,} |",
         f"| Agent cost (USD, list price, cache reads billed as full input) | {o['approx_cost_usd_upper_bound']:.2f} |",
         *([f"| Judge calls / cost (USD, upper bound) | {o['judge_usage']['calls']} / {o['judge_cost_usd_upper_bound']:.2f} |"]
           if "judge_usage" in o else []),
         "", "| Signal | TP | FP | FN | Precision | Recall |", "|---|---|---|---|---|---|"]
    for t in ALL_SIGNALS:
        x = s[t]
        L.append(f"| {t} | {x['tp']} | {x['fp']} | {x['fn']} | {pct(x['precision'])} | {pct(x['recall'])} |")
    L += ["", "| Feature (vs. golden-label features) | Median abs % err | Max abs % err |", "|---|---|---|"]
    for k, v in r["features"].items():
        L.append(f"| {k} | {v['median_abs_pct_err']:.2%} | {v['max_abs_pct_err']:.2%} |")
    L += ["", "| Judge negative control | Caught (judge failed the corrupted memo) |", "|---|---|"]
    for k, v in j["negative_control_catch_rate"].items():
        L.append(f"| {k} | {pct(v)} |")
    L += ["", f"Recommendation confusion (golden->predicted): `{rec['confusion']}`"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="data/golden")
    ap.add_argument("--backend", choices=["mock", "claude"], default="mock")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--judge", choices=["heuristic", "claude"], default="heuristic")
    ap.add_argument("--judge-model", default="claude-sonnet-5-5")
    ap.add_argument("--name", default=None)
    ap.add_argument("--reuse-run", action="store_true", help="score an existing runs/<name> without re-running")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    name = a.name or a.backend
    run_dir = Path("runs") / name
    if not a.reuse_run:
        run_pipeline(Path(a.golden) / "statements", run_dir, a.backend, a.model, a.effort, a.limit)
    judge = HeuristicJudge() if a.judge == "heuristic" else ClaudeJudge(a.judge_model)
    res = evaluate(Path(a.golden), run_dir, judge)
    Path("reports").mkdir(exist_ok=True)
    (Path("reports") / f"eval_{name}.json").write_text(json.dumps(res, indent=2, default=str))
    md = to_markdown(res)
    (Path("reports") / f"eval_{name}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
