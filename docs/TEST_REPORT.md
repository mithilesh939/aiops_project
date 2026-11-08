# Test report

> Numbers below come from a real run of the reference build (pytest 9.2, Python 3.13, Linux, CPU only).
> **Re-run on your machine and update this table before submitting** (`pytest -v`, then copy the counts from `reports/test_report.xml`).

| Suite | Tests | Passed | Failed | Skipped |
|---|---|---|---|---|
| tests/test_api.py | 13 | 13 | 0 | 0 |
| tests/test_data.py | 5 | 5 | 0 | 0 |
| tests/test_features.py | 5 | 5 | 0 | 0 |
| tests/test_models_drift.py | 9 | 9 | 0 | 0 |
| **Total** | **32** | **32** | **0** | **0** |

Run time 12.9 s. Lint: `ruff check src tests scripts` -> all checks passed.

## Acceptance criteria status
| ID | Status | Evidence |
|---|---|---|
| AC1 best AUC >= baseline + 0.02 | PASS (reference run: baseline 0.526, LR 0.758, LightGBM 0.760) | `reports/train_metrics.json`, MLflow |
| AC2 no leakage | PASS | TC-F3 |
| AC3 offline == online features | PASS | TC-F4 |
| AC4-AC6 API behaviour | PASS | TC-A1..A13 |
| AC7 drift only on drifted week | PASS in reference run: no alarm on batches 1-3 (PSI max 0.17); alarm on batch 4 (recent_acc5 PSI 0.42-0.51, question_difficulty 0.21-0.23; live AUC 0.78 -> 0.71) | monitor log, Grafana (your S23/S24) |
| AC8 retrain + promote | PASS in reference run: v2 promoted (hold-out AUC 0.736 vs 0.727); API served v2 without restart | MLflow, `/health` (your S27-S30) |
| AC9 metrics in Prometheus/Grafana | PASS for Prometheus: a real Prometheus 3.0.1 scraped the API and monitor (both targets UP), `promtool` validated config and the 5 alert rules, `DataDriftDetected` fired during the drift week and `ApiDown` fired when the API was stopped. All 23 dashboard queries were run against Prometheus and return data (the two error panels return a flat 0 when there are no errors). **Grafana itself could not be run in the build environment** - check that the dashboard renders on your machine (S24, S31) | Prometheus API, `promtool` |
| AC10 UI/API separation | PASS by construction (separate container, REST only) - UI rendering verified by JS syntax check only; **look at it in a browser** | |

## Not verified in the build environment (do these on your machine)
1. `docker compose up --build` (no Docker daemon / registry access there).
2. The Grafana dashboard rendering (Prometheus itself was tested).
3. The UI in a real browser (layout, mobile width).
4. GitHub Actions run.
