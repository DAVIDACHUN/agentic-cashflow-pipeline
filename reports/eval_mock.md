#### Backend `mock:rules-v1` | judge `heuristic-offline` | 40 statements, 6614 transactions

| Metric | Value |
|---|---|
| Classification accuracy | 95.4% |
| Classification macro-F1 | 0.935 |
| Routed to human review (conf < 0.8) | 5.6% |
| Accuracy on auto-accepted | 100.0% |
| Model accuracy on queued items | 18.0% |
| Accuracy after HITL (reviewers fix queued) | 100.0% |
| Risk-signal precision / recall (statement level) | 88.2% / 100.0% |
| Recommendation accuracy vs policy | 90.0% |
| Unsafe approvals (policy said refer/decline) | 0 |
| Judge pass rate | 45.0% |
| Mean turns / statement | 10.6 |
| Tool-call errors | 0 |
| Agent tokens in / out | 0 / 0 |
| Agent cost (USD, list price, cache reads billed as full input) | 0.00 |

| Signal | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| balance_tampering | 3 | 0 | 0 | 100.0% | 100.0% |
| duplicate_deposit | 2 | 0 | 0 | 100.0% | 100.0% |
| mca_stacking | 8 | 0 | 0 | 100.0% | 100.0% |
| nsf_cluster | 12 | 0 | 0 | 100.0% | 100.0% |
| pass_through_funds | 2 | 0 | 0 | 100.0% | 100.0% |
| structuring | 3 | 4 | 0 | 42.9% | 100.0% |

| Feature (vs. golden-label features) | Median abs % err | Max abs % err |
|---|---|---|
| avg_daily_balance | 0.00% | 0.00% |
| avg_monthly_revenue | 0.00% | 15.78% |
| avg_monthly_opex | 0.00% | 2.90% |
| revenue_volatility_weekly_cv | 0.00% | 284.29% |
| avg_monthly_debt_service | 2.34% | 33.33% |
| dscr_proxy | 15.60% | 496.43% |
| nsf_count | 0.00% | 0.00% |

| Judge negative control | Caught (judge failed the corrupted memo) |
|---|---|
| wrong_recommendation | 100.0% |
| inflated_metric | 100.0% |
| dropped_risk | 100.0% |

Recommendation confusion (golden->predicted): `{'approve->approve': 13, 'approve->decline': 3, 'approve->refer': 1, 'decline->decline': 8, 'refer->refer': 15}`
