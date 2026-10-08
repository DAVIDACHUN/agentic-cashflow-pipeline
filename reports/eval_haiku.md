#### Backend `claude:claude-haiku-5-5` | judge `claude:claude-sonnet-5-5` | 40 statements, 6614 transactions

| Metric | Value |
|---|---|
| Classification accuracy | 95.6% |
| Classification macro-F1 | 0.931 |
| Routed to human review (conf < 0.8) | 7.4% |
| Accuracy on auto-accepted | 100.0% |
| Model accuracy on queued items | 41.1% |
| Accuracy after HITL (reviewers fix queued) | 100.0% |
| Risk-signal precision / recall (statement level) | 100.0% / 93.3% |
| Recommendation accuracy vs policy | 90.0% |
| Unsafe approvals (policy said refer/decline) | 1 |
| Judge pass rate | 27.5% |
| Mean turns / statement | 6.5 |
| Tool-call errors | 0 |
| Agent tokens in / out | 4,092,614 / 364,956 |
| Agent cost (USD, list price, cache reads billed as full input) | 0.59 |
| Judge calls / cost (USD, upper bound) | 139 / 1.20 |

| Signal | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| balance_tampering | 3 | 0 | 0 | 100.0% | 100.0% |
| duplicate_deposit | 2 | 0 | 0 | 100.0% | 100.0% |
| mca_stacking | 8 | 0 | 0 | 100.0% | 100.0% |
| nsf_cluster | 12 | 0 | 0 | 100.0% | 100.0% |
| pass_through_funds | 2 | 0 | 0 | 100.0% | 100.0% |
| structuring | 1 | 0 | 2 | 100.0% | 33.3% |

| Feature (vs. golden-label features) | Median abs % err | Max abs % err |
|---|---|---|
| avg_daily_balance | 0.00% | 0.00% |
| avg_monthly_revenue | 0.00% | 15.78% |
| avg_monthly_opex | 0.77% | 16.89% |
| revenue_volatility_weekly_cv | 0.00% | 284.29% |
| avg_monthly_debt_service | 0.00% | 148.17% |
| dscr_proxy | 14.62% | 2904.76% |
| nsf_count | 0.00% | 0.00% |

| Judge negative control | Caught (judge failed the corrupted memo) |
|---|---|
| wrong_recommendation | 100.0% |
| inflated_metric | 100.0% |
| dropped_risk | 89.5% |

Recommendation confusion (golden->predicted): `{'approve->approve': 16, 'approve->refer': 1, 'decline->approve': 1, 'decline->decline': 5, 'decline->refer': 2, 'refer->refer': 15}`
