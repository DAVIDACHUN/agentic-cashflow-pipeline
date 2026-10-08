import json
import random
from datetime import date

import pytest

from cashflow_agent import detectors, review_queue
from cashflow_agent.agent import run_agent
from cashflow_agent.features import compute_features
from cashflow_agent.llm import MockBackend, Turn, rule_classify
from cashflow_agent.policy import recommend
from cashflow_agent.synth import generate_set, generate_statement
from cashflow_agent.tools import RunState, ToolError
from evals.judge import HeuristicJudge, corrupt


@pytest.fixture(scope="module")
def golden():
    return generate_set(20, seed=11)


def test_generator_deterministic():
    a, b = generate_set(5, 3), generate_set(5, 3)
    assert json.dumps(a) == json.dumps(b)


def test_running_balance_reconciles_unless_tampered(golden):
    for item in golden:
        s, g = item["statement"], item["golden"]
        tampered = any(x["type"] == "balance_tampering" for x in g["signals"])
        assert bool(detectors.balance_tampering(s, g["labels"])) == tampered


def test_detectors_recover_every_injected_signal_with_true_labels(golden):
    for item in golden:
        s, g = item["statement"], item["golden"]
        found = {c["type"] for c in detectors.run_all(s, g["labels"])}
        assert {x["type"] for x in g["signals"]} <= found


def test_features_basic_invariants():
    st = generate_statement(random.Random(0), 1, "stressed", date(2026, 2, 1))
    f = compute_features(st["statement"], st["golden"]["labels"])
    assert f["period_days"] == 90
    assert f["nsf_count"] == sum(v == "nsf_fee" for v in st["golden"]["labels"].values())
    assert f["avg_monthly_debt_service"] > 0 and f["mca_debit_count"] > 0


def test_policy():
    assert recommend({"structuring"}, {"dscr_proxy": 3}) == "decline"
    assert recommend({"nsf_cluster"}, {"dscr_proxy": 3}) == "refer"
    assert recommend(set(), {"dscr_proxy": 1.0}) == "refer"
    assert recommend(set(), {"dscr_proxy": None, "days_negative_balance": 0}) == "approve"


def test_tools_enforce_order_and_validate_ids(golden):
    st = RunState(golden[0]["statement"])
    with pytest.raises(ToolError):
        st.execute("compute_cashflow_features", {})
    out = st.execute("classify_transactions", {"items": [{"txn_id": "NOPE", "category": "rent", "confidence": 1}]})
    assert out["rejected"] == ["NOPE"]
    with pytest.raises(ToolError):
        st.execute("flag_risk", {"signal_type": "structuring", "txn_ids": ["NOPE"], "severity": "high", "evidence": ""})


def test_rule_classifier():
    assert rule_classify("NSF FEE", -35) == ("nsf_fee", 0.97)
    assert rule_classify("STRIPE TRANSFER 123", 500)[0] == "card_revenue"
    assert rule_classify("ELECTRONIC PAYMENT", -90)[1] < review_queue.CONF_THRESHOLD


def test_mock_agent_end_to_end_and_review_queue(golden, tmp_path):
    item = next(i for i in golden if i["golden"]["archetype"] == "fraud")
    res = run_agent(item["statement"], MockBackend())
    assert res.status == "completed" and res.tool_errors == 0
    assert len(res.state.classifications) == len(item["statement"]["transactions"])
    assert res.state.memo["recommendation"] == "decline"
    con = review_queue.connect(str(tmp_path / "q.db"))
    counts = review_queue.route_result(con, res)
    assert counts["memo"] == 1 and counts["flag"] >= 1


class BrokenBackend:
    """Simulates a model that calls a non-existent tool, then gives up."""
    name = "broken"

    def __init__(self):
        self.n = 0

    def step(self, system, tools, messages):
        self.n += 1
        if self.n == 1:
            b = [{"type": "tool_use", "id": "t1", "name": "drop_tables", "input": {}}]
            return Turn(b, b, "tool_use")
        b = [{"type": "text", "text": "cannot continue"}]
        return Turn(b, b, "end_turn")


def test_tool_errors_are_returned_not_raised_and_run_goes_to_review(golden, tmp_path):
    res = run_agent(golden[0]["statement"], BrokenBackend())
    assert res.tool_errors == 1 and res.status == "incomplete"
    con = review_queue.connect(str(tmp_path / "q.db"))
    assert review_queue.route_result(con, res)["statement"] == 1


def test_heuristic_judge_fails_negative_controls(golden):
    item = next(i for i in golden if i["golden"]["signals"])
    res = run_agent(item["statement"], MockBackend())
    judge = HeuristicJudge()
    for kind, bad in corrupt(res.state.memo, item["golden"]).items():
        assert not judge.judge(bad, item["golden"]).passed, kind


def test_dismissed_fraud_candidate_forces_human_review(golden, tmp_path):
    item = next(i for i in golden if any(s["type"] == "structuring" for s in i["golden"]["signals"]))
    stmt = item["statement"]
    labels = item["golden"]["labels"]
    rec = {"statement_id": stmt["statement_id"], "status": "completed",
           "classifications": {k: {"category": v, "confidence": 0.99} for k, v in labels.items()},
           "flags": [],  # agent dismissed everything
           "memo": {"recommendation": "approve", "summary": "", "key_risks": [], "cited_metrics": []},
           "candidates": detectors.run_all(stmt, labels)}
    con = review_queue.connect(str(tmp_path / "q.db"))
    counts = review_queue.route_record(con, rec, stmt)
    assert counts["flag"] >= 1 and counts["memo"] == 1
