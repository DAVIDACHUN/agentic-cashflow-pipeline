"""Offline wire test for the live Claude path.

A local HTTP server impersonates POST /v1/messages with canned responses, so
we exercise the real `anthropic` SDK end to end (request serialisation, typed
response parsing, appending SDK content blocks back into history, structured
output parsing) without credentials or network.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import anthropic
import pytest

from cashflow_agent.agent import run_agent
from cashflow_agent.llm import ClaudeBackend
from cashflow_agent.synth import generate_set
from evals.judge import ClaudeJudge


def _msg(content, stop):
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 20,
                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}


@pytest.fixture()
def fake_api():
    requests, script = [], []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})
            out = json.dumps(script.pop(0)).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    client = anthropic.Anthropic(api_key="test-key", base_url=f"http://127.0.0.1:{srv.server_port}", max_retries=0)
    yield client, requests, script
    srv.shutdown()


def test_agent_tool_round_trip_over_the_wire(fake_api):
    client, requests, script = fake_api
    stmt = generate_set(1, seed=5)[0]["statement"]
    script += [
        _msg([{"type": "text", "text": "Looking."},
              {"type": "tool_use", "id": "toolu_1", "name": "get_statement_overview", "input": {}}], "tool_use"),
        _msg([{"type": "text", "text": "Stopping here."}], "end_turn"),
    ]
    res = run_agent(stmt, ClaudeBackend(client=client))
    assert res.status == "incomplete"  # no memo submitted in this script
    first, second = requests[0]["body"], requests[1]["body"]
    assert requests[0]["path"] == "/v1/messages"
    assert first["model"] == "claude-haiku-5-5"
    assert first["output_config"] == {"effort": "low"}
    assert first["cache_control"] == {"type": "ephemeral"}
    assert all(t["strict"] and t["input_schema"]["additionalProperties"] is False for t in first["tools"])
    # history: assistant turn echoed back verbatim, tool_result in the next user turn
    assert second["messages"][1]["role"] == "assistant"
    assert second["messages"][1]["content"][1]["id"] == "toolu_1"
    tr = second["messages"][2]["content"][0]
    assert tr["type"] == "tool_result" and tr["tool_use_id"] == "toolu_1"
    assert json.loads(tr["content"])["statement_id"] == stmt["statement_id"]


def test_refusal_stops_the_run(fake_api):
    client, requests, script = fake_api
    stmt = generate_set(1, seed=5)[0]["statement"]
    script.append(_msg([], "refusal") | {"stop_details": {"type": "refusal", "category": None, "explanation": None}})
    assert run_agent(stmt, ClaudeBackend(client=client)).status == "refused"


def test_judge_structured_output_and_fallback_params(fake_api):
    client, requests, script = fake_api
    verdict = {"grounded": 5, "risk_coverage": 5, "recommendation_consistent": True, "unsupported_claims": [],
               "missing_risks": [], "passed": True, "rationale": "ok"}
    script.append(_msg([{"type": "text", "text": json.dumps(verdict)}], "end_turn") | {"model": "claude-sonnet-5-5"})
    item = generate_set(1, seed=5)[0]
    memo = {"recommendation": "approve", "summary": "x", "key_risks": [], "cited_metrics": []}
    v = ClaudeJudge(client=client).judge(memo, item["golden"])
    assert v.passed and v.grounded == 5
    body, headers = requests[0]["body"], requests[0]["headers"]
    assert body["model"] == "claude-sonnet-5-5"
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in headers.get("anthropic-beta", "")
    assert body["output_config"]["format"]["type"] == "json_schema"
