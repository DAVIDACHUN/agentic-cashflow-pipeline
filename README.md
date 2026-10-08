# agentic-cashflow-pipeline

An **agentic LLM pipeline for small-business bank-statement underwriting**, built on the Anthropic Claude API
with client-side tool use. The agent pages through a statement, classifies every transaction through tool
calls, reviews fraud/risk candidates from deterministic detectors, computes cash-flow features with code
(never by itself), and writes an underwriting memo. Anything uncertain or adverse goes to a **human-in-the-loop
review queue** (SQLite). The repo also includes an **eval harness**: a golden dataset, metric suite,
**LLM-as-a-judge** and negative controls that test the judge itself.

The whole pipeline runs **offline in mock mode with no API key**.

> **Data disclaimer.** Every statement is **synthetic** (`cashflow_agent/synth.py`). Business names,
> counterparties, amounts and the credit policy are invented for illustration. Nothing here is derived from a
> real lender, customer, employer or client system.

## Architecture

```
statement.json --> agent loop (cashflow_agent/agent.py)
                     |  Claude (claude-haiku-5-5, effort=low)  or  MockBackend (offline)
                     v
   tools (cashflow_agent/tools.py, strict JSON schemas)
     get_statement_overview / get_transactions(offset)          read
     classify_transactions([{txn_id, category, confidence}])    write labels
     run_anomaly_checks()      -> deterministic detectors  (detectors.py)
     flag_risk(type, txn_ids, severity, evidence)               agent confirms or dismisses candidates
     compute_cashflow_features() -> deterministic features (features.py)
     submit_underwriting_memo(recommendation, summary, risks, cited_metrics)
                     |
                     v
   review queue (review_queue.py, SQLite)  <-- confidence < 0.80, medium/high flags,
                                               DISMISSED fraud candidates, refer/decline memos,
                                               refused or failed runs
                     |
                     v
   evals/run_eval.py  -- golden labels, metrics, LLM judge (claude-sonnet-5-5) + negative controls
```

Design choices:
- **The LLM decides, code calculates.** Average daily balance, DSCR and the other features come from Python,
  so the memo can only quote numbers a tool returned.
- **Detectors are recall-first and the agent is the filter.** For example, the structuring detector fires on
  any three credits between $9,000 and $9,999 in 30 days. Legitimate B2B invoices land in that band too, and
  the agent's job is to read the descriptors and dismiss them.
- **Errors return to the model instead of crashing the run.** Unknown tools, bad transaction ids and calls made
  out of order come back as `is_error` tool results, and a run that ends without a memo is routed to review.
- **Claude API specifics:** strict tool schemas, every tool result returned in a single user turn, top-level
  prompt caching (the system prompt and tools form a stable prefix), `stop_reason` checks (`refusal` and
  `max_tokens` route to review), and structured outputs (Pydantic) for the judge with the server-side refusal
  fallback (`fallbacks="default"`).

## Synthetic golden set

40 statements of 90 days each and **6,614 transactions**, built from four archetypes (healthy / thin-margin /
stressed / fraud). Each statement carries:
- a category for every transaction (the same 12 categories as `finbert-transaction-classifier`);
- **injected signals**: `duplicate_deposit`, `balance_tampering` (running balance doctored mid-statement),
  `structuring`, `pass_through_funds`, `nsf_cluster` (NSF fees arise naturally from simulated balances),
  `mca_stacking` (two or more daily MCA debits);
- **confounders**: B2B contract wires followed by large supplier payments (these look like pass-through), and
  invoice payments inside the structuring band;
- the recommendation implied by an **illustrative credit policy** (`policy.py`).

## Results (measured)

Two agents on the same 40 statements:
- **Mock**: a deterministic scripted agent (`MockBackend`) with keyword rules that **accepts every detector
  candidate**, scored by the offline **heuristic** judge. Its rules were written against this generator's
  vocabulary, so its classification numbers are an in-sample upper bound for rules, not a performance claim.
- **Claude**: `claude-haiku-5-5` at `effort=low` as the agent, scored by a `claude-sonnet-5-5` judge (live API,
  October 2026, single run).

| Metric | Mock (offline rules) | Claude Haiku 5.5 (live) |
|---|---|---|
| Classification accuracy | 95.4% | 95.6% |
| Classification macro-F1 | 0.935 | 0.931 |
| Routed to human review (confidence < 0.80) | 5.6% | 7.4% |
| Accuracy on auto-accepted / on queued items | 100.0% / 18.0% | 100.0% / 41.1% |
| Risk-signal precision / recall (statement level) | 88.2% / 100.0% | **100.0% / 93.3%** |
| Structuring: TP / FP / FN | 3 / 4 / 0 | 1 / 0 / 2 |
| Recommendation accuracy vs. policy | 90.0% | 90.0% |
| Unsafe approvals (policy said refer/decline) | 0 | 1 (routed to a human under routing v2, see below) |
| Judge pass rate | 45.0% (heuristic judge) | 27.5% (Sonnet judge) |
| Mean turns per statement / tool-call errors | 10.6 / 0 | 6.5 / 0 |
| Mean latency per statement | - | 33 s |
| Agent tokens in / out | - | 4.09M / 0.36M |
| Cost (list price, upper bound: cache reads billed as full input) | $0 | agent $0.59 + judge $1.20 |

Recommendation confusion (golden → predicted):
- Mock: approve→decline 3, approve→refer 1, all others correct.
- Claude: approve→refer 1, decline→refer 2, **decline→approve 1**, all others correct.

Feature error caused by misclassification (agent-label features vs. golden-label features):

| Feature | Mock median / max abs % error | Claude median / max abs % error |
|---|---|---|
| avg_daily_balance | 0.00% / 0.00% | 0.00% / 0.00% |
| avg_monthly_revenue | 0.00% / 15.78% | 0.00% / 15.78% |
| avg_monthly_debt_service | 2.34% / 33.33% | 0.00% / 148.17% |
| dscr_proxy | 15.60% / 496.43% | 14.62% / 2,904.76% |

Judge reliability (negative controls: each memo is deliberately corrupted, and the judge must fail it):

| Corruption | Heuristic judge on mock memos | Sonnet judge on Claude memos |
|---|---|---|
| wrong recommendation | 100% | 100% (40/40) |
| average daily balance inflated 40% | 100% | 100% (40/40) |
| a true risk signal deleted | 100% | 89.5% (17/19), see note |

### What the numbers say

- **The LLM fixed the false positives and overcorrected.** Claude dismissed all 4 structuring false positives
  on legitimate B2B invoices (precision 42.9% → 100%), but it also dismissed 2 of the 3 *real* structuring
  patterns, reasoning that a cash-heavy business could plausibly make sub-$10k deposits. Recommendation
  accuracy is 90% for both agents, but the errors moved in the wrong direction: Claude's include
  **decline→approve**, the costliest kind of mistake.
- **The eval found a hole in the human-in-the-loop design.** In the unsafe approval (STMT-020), the agent
  noticed the four branch cash deposits and wrote "flagged for the human reviewer" in the memo, but it never
  raised a flag. Prose doesn't trigger routing, so under routing v1 that approve reached **no** human.
  Routing v2 (`review_queue.route_record`) treats dismissing a fraud-type detector candidate as a decision only a
  human can make. Every dismissed candidate is queued, and so is any approve memo that coexists with one.
  Re-routing the saved run (no new LLM calls; `pipeline --rebuild-queue`) puts both dismissed structuring
  cases and their memos in front of a reviewer, so 0 unsafe approvals now reach no human.
  `test_dismissed_fraud_candidate_forces_human_review` locks the rule in.
- **Small classification errors compound in ratio features.** One generic `ELECTRONIC PAYMENT` that is really
  a monthly loan payment moves DSCR a long way: the median DSCR error is about 15% for *both* agents even at
  about 95.5% transaction accuracy. Both agents turn the same healthy business into approve→refer because its
  DSCR comes out 1.02 instead of the true 1.41. This also drives most judge failures. It argues for
  computing coverage features only *after* low-confidence debits have been reviewed.
- **Confidence is informative for Claude.** Its auto-accepted labels (≥ 0.80) were 100.0% correct, while
  the 7.4% it routed to review were right only 41% of the time. With reviewers fixing the queue, the effective
  accuracy is 100%.
- **Test the judge, not just the agent.** Each problem below was found by running the eval, then fixed:
  - **Pilot run (3 statements):** the Sonnet judge passed 0/3. It had been given only the computed features,
    so it marked true details (transaction counts, closing balance, the policy threshold) as unsupported. It now
    sees the statement overview and the policy, and its rubric excludes counterparty-level detail it cannot verify.
  - **Wrong-recommendation control:** corrupting an already-wrong memo could accidentally make it right. Fixed by
    choosing a recommendation that differs from both the memo and the truth.
  - **Inflated-metric control:** a 1.4× balance happened to equal another metric. Fixed by checking structured
    citations against the same-named fact.
  - **Dropped-risk control:** deleting keywords leaked through paraphrase. It now deletes whole sentences using
    a synonym list. The 2 corrupted memos the Sonnet judge still passed were checked by hand: both still describe
    the risk in other words ("an unexplained +$12,833.65 offset"; "authenticity must be verified"), so the judge
    was right. Removing a risk from free text with rules has limits, and an LLM rewriter would make a stronger
    control.
- **The judge pass bar is strict on purpose.** A memo fails if any figure is more than 2% from ground truth,
  so a single misclassified transfer can sink it. The judge was not tuned toward passing; that would make the
  eval meaningless.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m cashflow_agent.synth                    # 40 synthetic statements + golden labels -> data/golden/
python -m evals.run_eval --backend mock --judge heuristic        # fully offline
pytest -q                                         # 14 tests incl. offline SDK wire tests

# human review queue
python -m cashflow_agent.review_queue --db runs/mock/review_queue.db stats
python -m cashflow_agent.review_queue --db runs/mock/review_queue.db list --limit 10
python -m cashflow_agent.review_queue --db runs/mock/review_queue.db resolve 7 --decision corrected \
    --reviewer da --correction '{"category": "loan_repayment"}'

# live (needs ANTHROPIC_API_KEY or an `ant auth login` profile); about $1.80 for the 40-statement run above
python -m evals.run_eval --backend claude --model claude-haiku-5-5 --judge claude --name haiku --limit 3   # pilot first
python -m evals.run_eval --backend claude --model claude-haiku-5-5 --judge claude --name haiku
python -m cashflow_agent.pipeline --out runs/haiku --rebuild-queue   # re-apply routing rules, no LLM calls
python -m evals.run_eval --backend claude --model claude-sonnet-5-5 --effort medium --judge claude --name sonnet
```

Outputs: `runs/<name>/results/*.json` (per-statement classifications, flags, features, memo, token usage, tool
trace), `runs/<name>/review_queue.db`, and `reports/eval_<name>.{json,md}`.

## Repo layout

```
cashflow_agent/synth.py         synthetic statements + golden labels
cashflow_agent/policy.py        illustrative credit policy (labels golden set, briefs the agent)
cashflow_agent/features.py      deterministic cash-flow features
cashflow_agent/detectors.py     deterministic fraud/risk candidate detectors
cashflow_agent/tools.py         tool schemas + executors (RunState)
cashflow_agent/llm.py           ClaudeBackend (Anthropic SDK) + MockBackend (offline)
cashflow_agent/agent.py         agentic loop + system prompt
cashflow_agent/review_queue.py  SQLite HITL queue + CLI
cashflow_agent/pipeline.py      batch runner
evals/run_eval.py               metric suite
evals/judge.py                  Claude judge (structured output) + heuristic judge + negative controls
tests/                          unit, end-to-end (mock) and wire tests
```

License: MIT.
