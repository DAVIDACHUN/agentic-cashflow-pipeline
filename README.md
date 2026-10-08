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
                                               refer/decline memos, refused or failed runs
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

> **Only offline mock mode has been run so far.** No Anthropic credentials were available when this was
> built, so the live Claude path has **not yet been scored on the golden set**. It is covered by an offline
> wire test (`tests/test_claude_wire.py`): a local fake Messages API checks request shape, the tool round-trip,
> refusal handling and judge structured-output parsing through the real `anthropic` SDK. Run the live eval
> command below to fill in the Claude column.

Mock backend = a deterministic scripted agent (`MockBackend`) with keyword rules that **accepts every detector
candidate**. Its rules were written against this generator's vocabulary, so its classification numbers are an
in-sample upper bound for rules and not a performance claim. It is in the repo as a transparent baseline and to
exercise the full pipeline. The judge here is the offline **heuristic** judge, which applies the same rubric
through string and number matching.

| Metric | Mock backend (offline) | Claude (`claude-haiku-5-5`) |
|---|---|---|
| Classification accuracy | 95.4% | not yet run |
| Classification macro-F1 | 0.935 | |
| Routed to human review (confidence < 0.80) | 5.6% | |
| Accuracy on auto-accepted / on queued items | 100.0% / 18.0% | |
| Risk-signal precision / recall (statement level) | 88.2% / 100.0% | |
| Structuring precision (confounder test) | 42.9% (3 TP, 4 FP) | |
| Recommendation accuracy vs. policy | 90.0% | |
| Unsafe approvals (policy said refer/decline) | 0 | |
| Judge pass rate | 45.0% | |
| Mean turns per statement / tool errors | 10.6 / 0 | |

Feature error caused by misclassification (agent-label features vs. golden-label features):

| Feature | Median abs % error | Max abs % error |
|---|---|---|
| avg_daily_balance | 0.00% | 0.00% |
| avg_monthly_revenue | 0.00% | 15.78% |
| avg_monthly_debt_service | 2.34% | 33.33% |
| dscr_proxy | 15.60% | 496.43% |

Judge reliability (negative controls: each memo is deliberately corrupted, and the judge must fail it):

| Corruption | Caught by heuristic judge |
|---|---|
| wrong recommendation | 100% |
| average daily balance inflated 40% | 100% |
| a true risk signal deleted | 100% |

### What the numbers say

- **Recall-first detectors need a smart filter.** Accepting every candidate (the mock) gets 100% recall but
  raises 4 structuring false positives. Three of them turn healthy B2B businesses into wrongful `decline`s,
  which is 3 of the 4 recommendation errors. The fourth landed on a statement already declined for a real
  fraud signal. Filtering these is the decision the live LLM is there to improve, and the eval isolates it.
- **Small classification errors compound in ratio features.** One generic `ELECTRONIC PAYMENT` that is really
  a monthly loan payment cuts debt service by a third and moves DSCR a long way: the median DSCR error is 15.6%
  even at 95% transaction accuracy. The remaining recommendation error comes from this: a healthy business's
  DSCR is computed as 1.02 instead of the true 1.41, below the policy's 1.25 line, so it becomes approve→refer.
  The same effect drives most of the 45% judge pass rate. It suggests computing coverage features only
  *after* low-confidence debits have been reviewed.
- **Test the judge, not just the agent.** The first version of the wrong-recommendation control was itself
  buggy: corrupting an already-wrong memo could accidentally make it right. A second issue was that a 1.4×
  balance happened to equal another metric. Both were found and fixed by running the controls. Free-text
  numbers are still matched against *any* fact value, which is a known limitation of the heuristic judge that
  the LLM judge does not share.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m cashflow_agent.synth                    # 40 synthetic statements + golden labels -> data/golden/
python -m evals.run_eval --backend mock --judge heuristic        # fully offline
pytest -q                                         # 13 tests incl. offline SDK wire tests

# human review queue
python -m cashflow_agent.review_queue --db runs/mock/review_queue.db stats
python -m cashflow_agent.review_queue --db runs/mock/review_queue.db list --limit 10
python -m cashflow_agent.review_queue --db runs/mock/review_queue.db resolve 7 --decision corrected \
    --reviewer da --correction '{"category": "loan_repayment"}'

# live (needs ANTHROPIC_API_KEY or an `ant auth login` profile)
python -m evals.run_eval --backend claude --model claude-haiku-5-5 --judge claude --name haiku
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
