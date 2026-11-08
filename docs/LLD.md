# QuizPulse - Low-Level Design (LLD)

Paradigm: functional core (pure functions for features, drift maths, splits) + a few small classes (`ModelHolder`, `KTModel`, SQLAlchemy models, pydantic schemas).
Python style: PEP 8 enforced by `ruff` (line length 120), docstrings on every module/public function, logging via `logging` (one format for all services).

## 1. REST API (FastAPI, base URL `http://localhost:8000`, interactive docs at `/docs`)

All bodies are JSON. Validation errors -> `422`. Unknown ids -> `404`. No model loaded -> `503`. Unexpected error -> `500` with a message (logged with stack trace).

### POST /predict
Request (`PredictRequest`)
| field | type | rules |
|---|---|---|
| student_id | string | 1-32 chars |
| question_id | string | 1-64 chars |
| skill | string | 1-64 chars |
| timestamp | float, optional | unix seconds; default = platform "now" |
| batch | int, optional | replay batch number, default -1 (live) |

Response `200` (`PredictResponse`): `prediction_id:int, student_id, question_id, skill, p_correct:float in [0,1], model_version:string, features:{overall_acc, skill_acc, skill_attempts, recent_acc5, hours_since_last, question_difficulty}`.
Side effects: row in `predictions`; metrics `qp_predictions_total`, `qp_prediction_value`.

```
POST /predict {"student_id":"S0001","question_id":"FRA_03","skill":"Fractions"}
-> 200 {"prediction_id":1,"p_correct":0.41,"model_version":"1","features":{...}}
```

### POST /outcomes
Request (`OutcomeRequest`): `correct:int (0|1)` **and** either `prediction_id:int` **or** (`student_id`,`question_id`,`skill` [+ `timestamp`]).
Response `200`: `{"status":"recorded","attempt_id":int}`. `404` if `prediction_id` unknown, `422` if identity is missing or `correct` not 0/1.
Side effects: row in `outcomes` (delayed label) and `attempts` (history grows -> later features change).

### GET /students/{student_id}/mastery  and  GET /students/{student_id}/weak-topics[?include_strong=true]
Response `WeakTopicsResponse`: `student_id, threshold (0.5), model_version, topics:[{skill, mastery, attempts, observed_accuracy|null, weak}]`, **sorted by mastery ascending**.
`weak-topics` returns only topics with mastery < 0.5. Mastery of a topic = mean predicted `p_correct` over up to 6 known questions of the topic, evaluated "now".

### POST /next-questions
Request: `{"student_id": str, "n": int 1..20 (default 5)}`. Response: `{student_id, model_version, questions:[{question_id, skill, p_correct, reason}]}`.
Algorithm: weakest 3 topics -> unseen questions -> prefer predicted p closest to 0.6 -> spread over the topics.

### Lookups
`GET /students?q=<prefix>&limit=<n>` -> `[{student_id, attempts}]`; `GET /skills` -> `[str]`; `GET /questions?skill=<s>` -> `[question_id]`.

### Operations
| Endpoint | Purpose | Response |
|---|---|---|
| `GET /health` | liveness + readiness | `{status: ok|degraded, model_loaded, model_version, database}` |
| `GET /metrics` | Prometheus exposition | text |
| `GET /model/info` | registry versions with aliases, AUC, git commit, stage | `{model_name, serving_version, loaded_at, versions[]}` |
| `GET /pipeline/runs?kind=&limit=` | audit trail (ingest, train, register, drift, retrain, promotion) | list |
| `GET /pipeline/status` | counts per stage/status + average duration | dict |
| `GET /pipeline/throughput` | data-quality + feature-stage speed from `reports/*.json` | dict |
| `GET /monitoring/summary` | counts, rolling metrics, last drift run, serving version | dict |
| `GET /reports/drift` | latest Evidently HTML | text/html |
| `POST /admin/reload` | reload production model now | `{reloaded, serving_version}` |
| `POST /admin/drift-check` | forwards to monitor `/drift/run` | drift result dict |
| `POST /admin/retrain` | forwards to monitor `/retrain/run` (forced) | retrain result dict |

### Monitor service (`:8100`)
`GET /metrics`, `GET /health`, `POST /drift/run`, `POST /retrain/run` (409 if one is already running), `GET /reports/drift`.

### MLflow Serve (optional, `:5001`)
`POST /invocations` with `{"dataframe_records":[{overall_acc, skill_acc, skill_attempts, recent_acc5, hours_since_last, question_difficulty, skill}]}` -> `{"predictions":[p]}`; wrong columns -> `400` (signature validation).

## 2. Data model (PostgreSQL / SQLite)
| Table | Columns |
|---|---|
| attempts | id PK, user_id (idx), skill, question_id, correct, ts (float secs), batch; idx (question_id, ts), (user_id, ts) |
| predictions | id PK, user_id, skill, question_id, p_correct, model_version, ts, batch, 6 feature columns |
| outcomes | id PK, prediction_id (idx), user_id, correct, ts |
| pipeline_runs | id PK, kind, status (running/success/failed/skipped), started, finished, duration_s, rows, details (JSON text), mlflow_run_id |

The `attempts` table has the schema of a quiz-attempt log, so a real Sanrathi backend could feed it directly (future work).

## 3. Model contract (MLflow pyfunc)
Input DataFrame: 6 float features + `skill` (string). Output: numpy array of P(correct), clipped to [1e-6, 1-1e-6]. `KTModel` wraps `SkillMeanBaseline`, a scikit-learn pipeline (LR) or `LGBMClassifier`.
Logged with `signature`, `input_example`, `code_paths=[src]` (so it loads in any container) and pinned pip requirements.

## 4. Modules and key functions
| Module | Functions / classes |
|---|---|
| `common/config.py` | `load_params()`, `Settings` (env), `setup_logging()` |
| `common/db.py` | SQLAlchemy models, `session_scope()`, `wait_for_db()`, `log_run()` |
| `data/synthetic_generator.py` | `generate(...)` |
| `data/ingest.py` | `load_assistments()`, `clean()`, `validate()` |
| `data/make_batches.py` | `assign_batches()` |
| `features/build_features.py` | `user_features()`, `add_question_difficulty()`, `build_features(engine)`, `online_features()` |
| `models/train.py` | `train_one()`, `build_estimator()`; `models/evaluate.py`: `evaluate()`, `time_split()` |
| `models/register.py` | `register_run()`, `promote()`, `register_best()`, `production_version()` |
| `serving/app.py` | endpoints + `instrument` middleware (metrics + access log) |
| `serving/model_loader.py` | `ModelHolder.load_from_registry()`, `.watch()` (hot reload) |
| `serving/mastery.py` | `student_mastery()`, `next_questions()`, `platform_now()` |
| `monitoring/performance_monitor.py` | `rolling_metrics()` (join predictions + outcomes) |
| `monitoring/drift.py` | `psi()`, `compute_drift()`, `check()`, `evidently_html()` |
| `monitoring/monitor_app.py` | background loop, gauges, `/drift/run`, `/retrain/run` |
| `retrain/retrain.py` | `split_for_retrain()`, `retrain()` |

## 5. Feature definitions (all past-only)
| Feature | Definition | Cold start |
|---|---|---|
| overall_acc | mean correctness of all previous attempts of the student | 0.5 |
| skill_acc | same, on this skill | overall_acc |
| skill_attempts | number of previous attempts on this skill | 0 |
| recent_acc5 | mean of the last 5 previous attempts | overall_acc |
| hours_since_last | hours since previous attempt, capped at 720 | 720 |
| question_difficulty | (sum_correct + 0.5*10) / (n + 10) of the question over all earlier attempts | 0.5 |

## 6. Prometheus metrics
API: `qp_http_requests_total{method,endpoint,status}`, `qp_http_request_seconds{endpoint}` (histogram), `qp_predictions_total{model_version}`, `qp_prediction_value` (histogram of p),
`qp_outcomes_total{correct}`, `qp_errors_total{endpoint,type}`, `qp_model_version`, `qp_model_loaded`, `qp_model_reloads_total`, `qp_feature_build_seconds`, `qp_build_info`.
Monitor: `qp_rolling_auc|logloss|accuracy|labeled_rows`, `qp_feature_psi{feature}`, `qp_prediction_psi`, `qp_drift_detected`, `qp_auc_drop`, `qp_retrain_runs_total{result}`, `qp_last_drift_check_timestamp`.
Alert rules (`docker/prometheus/rules.yml`): ApiDown, HighLatencyP95, DataDriftDetected, RollingAucLow, ErrorRateHigh.

## 7. Error handling and logging
* Every endpoint validates input (pydantic) and raises `HTTPException` with a precise status; unexpected exceptions are logged with a stack trace, counted in `qp_errors_total` and returned as 500.
* The model loader never crashes the API: if the registry has no production model yet, `/health` reports `degraded`, `/predict` returns 503 and the watcher keeps polling.
* The monitor loop catches and logs every exception so one bad check cannot stop monitoring; a retrain already in progress returns 409.
* Log format: `time | level | component | message`; access log line per request with latency.

## 8. Configuration
`params.yaml` (all numbers: generator, batches, features, training hyper-parameters, drift thresholds, retrain gate) and environment variables: `DATABASE_URL`, `MLFLOW_TRACKING_URI`, `MODEL_NAME`, `MODEL_ALIAS`,
`MODEL_POLL_SECONDS`, `MONITOR_URL`, `MONITOR_INTERVAL`, `AUTO_RETRAIN`, `REPORTS_DIR`, `CORS_ORIGINS`, `LOG_LEVEL`. UI: `frontend/config.js`.
