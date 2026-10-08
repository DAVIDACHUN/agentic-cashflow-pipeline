"""LLM-as-a-judge for underwriting memos, plus an offline heuristic stand-in.

The judge sees the memo and the GROUND-TRUTH facts (golden features, injected
signals, policy recommendation) and grades:
  grounded (1-5)               - every number in the memo matches a fact
  risk_coverage (1-5)          - every true risk signal is mentioned
  recommendation_consistent    - recommendation matches the policy outcome
  passed                       - grounded >= 4, full risk coverage, consistent

Judge reliability is itself tested with NEGATIVE CONTROLS: each memo is
corrupted (wrong recommendation / inflated metric / dropped risk) and a good
judge must fail the corrupted versions. See `corrupt()`.
"""
from __future__ import annotations

import copy
import json
import re

from pydantic import BaseModel

SIGNAL_WORDS = {
    "duplicate_deposit": ["duplicate"],
    "balance_tampering": ["tamper", "balance discontinu", "doctored", "altered"],
    "structuring": ["structur"],
    "pass_through_funds": ["pass-through", "pass through", "pass_through"],
    "nsf_cluster": ["nsf"],
    "mca_stacking": ["mca", "merchant cash advance", "stacking"],
}


class JudgeVerdict(BaseModel):
    grounded: int
    risk_coverage: int
    recommendation_consistent: bool
    unsupported_claims: list[str]
    missing_risks: list[str]
    passed: bool
    rationale: str


JUDGE_SYSTEM = """You are a strict reviewer of small-business underwriting memos. You are given \
GROUND-TRUTH FACTS and a MEMO written by an automated agent. Grade the memo only against the facts.

- grounded (1-5): 5 = every number in the memo matches a fact value within 2% relative; subtract \
for each number that does not match or is not in the facts. Rounding/formatting differences are fine.
- risk_coverage (1-5): 5 = every ground-truth risk signal is clearly mentioned; 1 = none are. \
If there are no ground-truth signals, score 5 unless the memo invents serious risks.
- recommendation_consistent: true only if the memo's recommendation equals the policy recommendation.
- unsupported_claims: numbers or claims not supported by the facts.
- missing_risks: ground-truth signal types the memo fails to mention.
- passed: true only if grounded >= 4 AND risk_coverage == 5 AND recommendation_consistent."""


def facts_for(golden: dict) -> dict:
    return {"features": golden["features"],
            "risk_signals": sorted({s["type"] for s in golden["signals"]}),
            "policy_recommendation": golden["recommendation"]}


def memo_text(memo: dict) -> str:
    return json.dumps(memo, indent=1)


class ClaudeJudge:
    """Live judge: claude-sonnet-5-5 with structured output (Pydantic)."""

    def __init__(self, model: str = "claude-sonnet-5-5", effort: str = "medium", client=None):
        import anthropic

        self.client = client or anthropic.Anthropic()
        self.model, self.effort = model, effort
        self.name = f"claude:{model}"

    def judge(self, memo: dict, golden: dict) -> JudgeVerdict | None:
        resp = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=4000,
            system=JUDGE_SYSTEM,
            output_config={"effort": self.effort},
            output_format=JudgeVerdict,
            # server-side refusal fallback: if a safety classifier declines, the API reroutes
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content":
                       f"GROUND-TRUTH FACTS:\n{json.dumps(facts_for(golden), indent=1)}\n\nMEMO:\n{memo_text(memo)}"}],
        )
        if resp.stop_reason == "refusal":
            return None
        return resp.parsed_output


class HeuristicJudge:
    """Offline, deterministic stand-in with the same rubric. Not an LLM: it
    string-matches numbers and risk keywords. Used when no API key is set."""

    name = "heuristic-offline"
    _num = re.compile(r"-?\$?\d[\d,]*\.?\d*")

    def judge(self, memo: dict, golden: dict) -> JudgeVerdict:
        facts = facts_for(golden)
        fact_vals = [float(v) for v in facts["features"].values() if isinstance(v, (int, float))]

        def supported(x: float) -> bool:
            return any(abs(x - f) <= max(0.02 * abs(f), 0.011) for f in fact_vals)

        feats = facts["features"]
        claims = []
        bad_named = []  # structured citations must match the SAME-named fact, not just any fact
        for m in memo.get("cited_metrics", []):
            f = feats.get(m["name"])
            if isinstance(f, (int, float)) and abs(m["value"] - f) > max(0.02 * abs(f), 0.011):
                bad_named.append(round(float(m["value"]), 2))
            elif not isinstance(f, (int, float)):
                claims.append(m["value"])
        for tok in self._num.findall(memo.get("summary", "")):
            try:
                claims.append(float(tok.replace("$", "").replace(",", "").rstrip(".")))
            except ValueError:
                pass
        # free-text numbers: matched against ANY fact value (known limitation, see README)
        bad = sorted(set(bad_named) | {round(float(c), 2) for c in claims if not supported(float(c))})
        grounded = max(1, 5 - 2 * len(bad))

        text = (memo.get("summary", "") + " " + " ".join(memo.get("key_risks", []))).lower()
        missing = [s for s in facts["risk_signals"]
                   if s not in text and not any(w in text for w in SIGNAL_WORDS[s])]
        n = len(facts["risk_signals"])
        coverage = 5 if n == 0 else 1 + round(4 * (n - len(missing)) / n)
        consistent = memo.get("recommendation") == facts["policy_recommendation"]
        passed = grounded >= 4 and coverage == 5 and consistent
        return JudgeVerdict(grounded=grounded, risk_coverage=coverage, recommendation_consistent=consistent,
                            unsupported_claims=[str(b) for b in bad], missing_risks=missing, passed=passed,
                            rationale="heuristic rubric check")


def corrupt(memo: dict, golden: dict) -> dict[str, dict]:
    """Return named corrupted copies of a memo (negative controls)."""
    out = {}
    m = copy.deepcopy(memo)
    # must differ from BOTH the memo and the truth, otherwise "corrupting" a wrong memo could fix it
    others = [r for r in ("approve", "refer", "decline") if r not in (memo["recommendation"], golden["recommendation"])]
    m["recommendation"] = others[0]
    m["summary"] = re.sub(r"Recommendation: \w+", f"Recommendation: {others[0]}", m["summary"])
    out["wrong_recommendation"] = m

    m = copy.deepcopy(memo)
    adb = golden["features"]["avg_daily_balance"]
    for c in m.get("cited_metrics", []):
        if c["name"] == "avg_daily_balance":
            c["value"] = round(c["value"] * 1.4, 2)
    m["summary"] = f"Average daily balance is strong at ${adb * 1.4:,.2f}. " + m["summary"]
    out["inflated_metric"] = m

    sigs = sorted({s["type"] for s in golden["signals"]})
    if sigs:
        drop = sigs[0]
        m = copy.deepcopy(memo)
        m["key_risks"] = [r for r in m.get("key_risks", [])
                          if drop not in r.lower() and not any(w in r.lower() for w in SIGNAL_WORDS[drop])]
        for w in [drop] + SIGNAL_WORDS[drop]:
            m["summary"] = re.sub(re.escape(w), "", m["summary"], flags=re.I)
        out["dropped_risk"] = m
    return out
