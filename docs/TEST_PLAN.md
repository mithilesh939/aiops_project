# QuizPulse - Test plan, test cases and acceptance criteria

## 1. Scope and strategy
| Level | What | How |
|---|---|---|
| Unit | feature maths, metrics, baseline, PSI, splits, generator, validation | `pytest` (pure functions) |
| Integration | API + SQLite + feature code + (fake) model; Ray vs pandas equality | `pytest` with FastAPI `TestClient`, fake model |
| System / end-to-end | Compose stack or local processes: replay -> drift -> auto-retrain -> new model served | manual run with the demo script (`docs/STEP_BY_STEP_GUIDE.md` section 8), evidence = screenshots S23-S31 |
| Static | lint | `ruff check src tests scripts` |

Tests run in CI (`.github/workflows/ci.yml`) and as the DVC stage `test` (`reports/test_report.xml`).
Test isolation: `tests/conftest.py` points `DATABASE_URL`/`MLFLOW_TRACKING_URI` to a temp directory and injects a deterministic fake model, so tests need no network, MLflow or Docker.

## 2. Acceptance criteria
| ID | Criterion | Verified by |
|---|---|---|
| AC1 | Best model AUC >= baseline AUC + 0.02 on the time-ordered validation set | `register.py` refuses to register otherwise; MLflow compare (S07) |
| AC2 | No leakage: features of a row never depend on its own or later labels | TC-F3 |
| AC3 | Online features equal offline features (no train/serve skew) | TC-F4 |
| AC4 | `/predict` returns p in [0,1], model version, features; invalid input -> 422; no model -> 503 | TC-A2..A3, A11 |
| AC5 | Outcomes are stored and extend the student's history | TC-A5, A8 |
| AC6 | Weak topics are sorted by mastery and below 0.5 | TC-A9 |
| AC7 | No drift alarm on normal weeks; drift alarm on the injected drift week | System test ST2/ST3 (replay), TC-D5/D6 |
| AC8 | A drift triggers a retrain; a better challenger is promoted and served without restart; a worse one is rejected | System test ST4/ST5, TC-R1 |
| AC9 | Metrics exposed to Prometheus and visible in Grafana within 15 s | TC-A12, ST6 |
| AC10 | UI and API are separate services connected only by REST | Compose file, `frontend/config.js` |

## 3. Automated test cases (32)
| ID | Test | Expected |
|---|---|---|
| TC-F1 | features_have_no_nulls_and_valid_ranges | no NaN, accuracies in [0,1], counts >= 0 |
| TC-F2 | first_attempt_uses_prior_not_label | first attempt: overall_acc=0.5, skill_attempts=0 |
| TC-F3 | no_leakage_a_label_only_affects_the_future | flipping one label changes no feature at or before that time |
| TC-F4 | offline_online_parity | 60 random rows: online features == offline features (1e-9) |
| TC-F5 | ray_matches_pandas | identical features from both engines |
| TC-D1 | generator_is_deterministic | same seed -> identical frame |
| TC-D2 | drift_week_introduces_new_skills_and_harder_questions | new skills appear; accuracy drops in drift week |
| TC-D3 | validate_passes_on_good_data | report.passed |
| TC-D4 | validate_rejects_non_binary_target | ValueError |
| TC-D5b | batches_are_time_ordered_and_disjoint | batch k ends before batch k+1 starts |
| TC-M1 | evaluate_perfect_and_random | AUC 1.0 and 0.5 |
| TC-M2 | evaluate_single_class_gives_nan_auc | NaN AUC, no crash |
| TC-M3 | time_split_is_chronological | train max ts <= valid min ts, 20% valid |
| TC-M4 | baseline_predicts_skill_mean_and_handles_unseen_skill | skill mean; global mean for unseen |
| TC-M5 | kt_wrapper_returns_probabilities | shape (n,), 0<p<1 |
| TC-D5 | psi_is_zero_for_same_distribution_and_large_for_shift | <0.05 vs >0.25 |
| TC-D5c | psi_binary_feature | categorical PSI works for 0/1 indicator |
| TC-D6 | compute_drift_flags_shifted_features | shifted -> dataset_drift; same distribution -> no drift |
| TC-R1 | retrain_split_holdout_is_strictly_newer | hold-out later than training rows |
| TC-A1 | health_ok | 200, model_loaded, database true |
| TC-A2 | predict_returns_probability_and_features | 200, p in [0,1], 6 features |
| TC-A3 | predict_validates_input | 422 for empty id / missing fields |
| TC-A4 | predict_new_student_cold_start | works, skill_attempts=0 |
| TC-A5 | outcome_roundtrip_adds_attempt | 200 recorded |
| TC-A6 | outcome_unknown_prediction_is_404 | 404 |
| TC-A7 | outcome_requires_valid_label_and_identity | 422 |
| TC-A8 | outcome_without_prediction_id_but_full_identity | 200 |
| TC-A9 | weak_topics_sorted_ascending | sorted; weak < 0.5 |
| TC-A10 | next_questions | 1..n suggestions; n=0 -> 422 |
| TC-A11 | 503_when_no_model_loaded | 503 and /health degraded |
| TC-A12 | metrics_endpoint_exposes_business_metrics | `qp_*` series present |
| TC-A13 | pipeline_endpoints | runs/status/summary respond |

## 4. System test scenarios (manual, with evidence)
| ID | Steps | Expected | Evidence |
|---|---|---|---|
| ST1 | `docker compose up --build`; open UI | all services healthy, UI shows "API connected" | S18, S32 |
| ST2 | replay batches 1-3 | no drift, live AUC ~ training AUC, PSI < 0.2 | S23 |
| ST3 | replay batch 4 (drift) | `qp_drift_detected`=1, AUC drop, Evidently report | S24-S26 |
| ST4 | automatic retrain | new model version registered; promoted when better | S27, S28 |
| ST5 | `/health`, `/predict` | serves the new version without restart | S30 |
| ST6 | Grafana during ST2-ST4 | panels update within 15 s | S23, S24, S31 |
| ST7 | stop `api` container | `ApiDown` alert pending/firing in Prometheus | S22 |
| ST8 | UI: a student id that has no history | cold-start topics (0 attempts, prior-based mastery), no crash | S32 |

## 5. Test report
See `docs/TEST_REPORT.md` (regenerate it from your own run: `pytest -v`, `reports/test_report.xml`).
