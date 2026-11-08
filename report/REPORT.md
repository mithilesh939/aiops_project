# QuizPulse: an AIOps lifecycle for knowledge tracing

*Course project - AI Operations. All numbers are from a real run of this repository (reference build); replace them with your own runs.*

## 1. Problem
Quiz platforms (e.g. Sanrathi) store every attempt but do not turn it into guidance. QuizPulse predicts P(correct) for a student's next question (binary classification / knowledge tracing),
aggregates it into per-topic mastery, flags weak topics (< 0.5) and recommends the next questions. Success = best model beats the baseline by >= 0.02 AUC **and** the system stays healthy under change (drift detection + automatic, gated retraining).

## 2. Data
| Item | Value |
|---|---|
| Source used in the reference run | synthetic Sanrathi-style generator (seeded, reproducible); ASSISTments 2009-10 supported via `data.source: assistments` (check licence) |
| Size | 50,454 attempts, 600 students, 15 topics, 375 questions, 7 simulated weeks |
| Batches | batch_0 = weeks 1-3 (21,554 rows, training), batch_1..4 = weeks 4-7 (live replay) |
| Drift injected in week 7 | 3 new syllabus topics (50% of attempts) + every question 1.0 logit harder |
| Quality checks | schema, nulls, binary target, time ordering, min rows -> `reports/data_quality.json` |
Split is **by time**; features use **only past attempts** (tested).

## 3. Models and results (MLflow experiment `quizpulse-training`)
Features: overall accuracy, topic accuracy, attempts on topic, accuracy over last 5, hours since last attempt, question difficulty (smoothed historical correct rate).
| Model | Valid AUC | Valid log-loss | Valid acc | Test AUC (next week) |
|---|---|---|---|---|
| Baseline (skill mean) | 0.526 | 0.689 | 0.549 | 0.523 |
| Logistic regression | 0.758 | 0.588 | 0.689 | 0.762 |
| LightGBM (registered) | **0.760** | **0.583** | **0.692** | **0.768** |
The tree model and the linear model tie because the synthetic response is close to logistic; personalisation (any student-level feature) is what lifts AUC from 0.53 to 0.76.

## 4. MLOps design
(see `docs/HLD.md`, `docs/LLD.md`, `docs/architecture.png`)
* **Versioning:** Git (code), DVC DAG `generate -> ingest -> batches -> features(Ray) -> train -> test` with `params.yaml`, MLflow (runs, models; tags `git_commit`, `data_version`).
* **Tracking and registry:** 3 models compared; params, validation/test metrics, ROC and importance plots, drift reference sample, config and a signed pyfunc model are logged; alias `production` marks the served version.
* **Serving:** FastAPI (`/predict`, `/outcomes`, `/students/{id}/weak-topics`, `/next-questions`, ...), model hot-reloaded from the registry; MLflow Serve exposes the same model; separate nginx UI container; Postgres store.
* **Monitoring:** Prometheus scrapes the API (rate, latency histogram, errors, prediction distribution, model version) and the monitor (rolling AUC, PSI per feature, drift flag); Grafana dashboard (15 panels) and 5 alert rules.
* **Drift + retraining:** PSI-based decision (dataset drift = >= 30% features with PSI > 0.2, prediction drift, or live AUC drop > 0.04) + Evidently HTML; on detection the retrain job trains a challenger, compares it with production on the same unseen time-ordered hold-out and promotes only if better.
* **Packaging/CI:** Docker Compose (frontend and api as separate services), DVC DAG, GitHub Actions (lint, 32 tests, pipeline + acceptance gate, image builds).
* **Data engineering throughput:** ingest ~670k rows/s (validation + write); feature stage 50,454 rows: pandas ~28k rows/s, Ray ~3.5k rows/s (Ray start-up dominates at this size).

## 5. Drift experiment
Replay of batches 1-3 (40% of students, all their rows) then the drifted batch 4:
| Window | Live AUC | Log-loss | Highest PSI | Drift flag |
|---|---|---|---|---|
| Normal weeks (1500 labelled predictions) | 0.781 | 0.553 | skill_acc 0.17 (< 0.2) | no |
| Drifted week | 0.711 | 0.629 | recent_acc5 0.42-0.51, question_difficulty 0.21-0.23 | **yes** |
Detection reasons logged: "feature drift: 2/6 features PSI > 0.2" and "rolling AUC dropped 0.048 (> 0.04)". The monitor then triggered the retrain automatically.
Retrain: challenger (LightGBM on all data incl. the first 60% of the recent window) scored on the last 40% of the recent window: AUC **0.736 vs 0.727** for the old model, log-loss 0.602 vs 0.622 -> promoted as v2; the API served v2 within seconds with no restart; after the retrain the reference became the new regime and the drift flag cleared.
A later forced retrain with no further change produced a tie and is rejected by the gate (+0.002 AUC required).

## 6. Lessons and problems found while building
* A first drift design flagged drift on normal weeks: cumulative features (counts, running accuracy) grow as histories mature. Fixed by (a) monitoring the stationary `cold_start` indicator instead of the raw counter, (b) taking the reference from the matured part of the training data, (c) letting the retrain define a new reference.
* Replaying only the first N rows of a week (instead of all rows of a subset of students) produced unrealistic `hours_since_last` -> replay uses a student subset (`--student-fraction`).
* Pandas 3 broke Evidently 0.7 -> pinned `pandas<3`.
* Promotion on ties would churn versions -> minimum AUC gain 0.002 plus a log-loss condition.

## 7. Limitations and future work
Synthetic data and injected drift (the pipeline also accepts ASSISTments); gain from retraining is small and measured on ~2,400 hold-out rows; single-node Compose, no auth/TLS/encryption; Ray gives no speed-up at this data size; Alertmanager not wired (rules only);
no Kubernetes/Kafka. Future: connect the `attempts` table to Sanrathi's backend (same schema), Deep Knowledge Tracing (GRU), calibration, canary/shadow deployment before promotion.

## 8. AI-use disclosure
See `AI_DISCLOSURE.md`.
