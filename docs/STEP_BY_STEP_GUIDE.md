# QuizPulse - Step-by-step guide (what to run, what to screenshot, which lecture explains it)

Course repo (your professor's): <https://github.com/sudarsun/DA3408>
Lecture links below are relative to `https://github.com/sudarsun/DA3408/blob/main/`.

> **Read this first - AI-use rules.** The course's `AI-fair-use-code-of-conduct.pdf` says you must be able to explain
> every line you submit (the "whiteboard test"), must disclose AI use (`AI_DISCLOSURE.md` is included - edit it so it
> is true for *you*), and must **never fabricate metrics, logs or Grafana screenshots**. Every screenshot below must
> come from *your own* run on *your own* machine. Do the steps, read the code of each file as you go, and use
> `docs/VIVA_PREP.md` to rehearse.

---------------------------------------------------------------------------------------------------

## 0. What is graded and where it lives in this project

| Rubric item (AI Application Evaluation Guideline) | Pts | Where in QuizPulse |
|---|---|---|
| Web UI/UX | 3 | `frontend/` (student view + manual in `docs/USER_MANUAL.md`) |
| ML pipeline visualisation | 2 | UI tab **Pipeline & models** + MLflow + Grafana + DVC DAG |
| Design principle (HLD, LLD, diagrams, loose coupling) | 2 | `docs/HLD.md`, `docs/LLD.md`, `docs/architecture.png`; UI <-> API only via REST (`frontend/config.js`) |
| Implementation (style, logging, exceptions, comments, unit tests) | 2 | `src/`, `ruff`, `tests/` |
| Testing (plan, cases, report, acceptance) | 1 | `docs/TEST_PLAN.md`, `reports/test_report.xml` |
| Data engineering (Ray/Spark/K8s) | 2 | `src/features/build_features.py` (Ray Data), throughput in `reports/feature_stats.json` |
| Source control & CI with DVC | 2 | `dvc.yaml` DAG, Git, `.github/workflows/ci.yml` |
| Experiment tracking | 2 | `src/models/train.py` (MLflow) |
| Prometheus + Grafana | 2 | `src/serving/metrics.py`, `src/monitoring/monitor_app.py`, `docker/` |
| Packaging (MLflow serve, MLproject, FastAPI, Docker, Compose 2 services) | 2 | `MLproject`, `src/serving/app.py`, `docker-compose.yml` |
| Viva | 5 | `docs/VIVA_PREP.md` |

Mandatory documents (5): architecture diagram, HLD, LLD, test plan + cases, user manual -> all in `docs/`.

---------------------------------------------------------------------------------------------------

## QUICKEST PATH - one script, no Docker (do this first, about 8 minutes)

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt
bash scripts/demo_run.sh
```
It prints labelled sections (S01, S02, S03, S04, S05, S12, S13, S14, S15, S17, S23, S24, S25, S28, S29, S30). **Screenshot your own terminal at each one.**
When it ends the services keep running, so take the browser screenshots at:
- http://localhost:8080 - the UI (both tabs, plus a narrow window)
- http://localhost:5000 - MLflow (S07-S10, S27, S28)
- http://localhost:8000/docs - S11
- http://localhost:8000/reports/drift - S26

Grafana/Prometheus screenshots (S20-S24, S31) need `docker compose up --build` (section 7). Stop the local services first with the command printed at the end of the script, because the ports clash.

## 1. Set up your own copy (Day 1)

```bash
# put the project in a NEW folder that is yours
cp -r quizpulse ~/quizpulse && cd ~/quizpulse
python3 -m venv .venv && source .venv/bin/activate     # Python 3.10-3.12 recommended
pip install -r requirements-dev.txt
git init -b main && dvc init
git add -A && git commit -m "chore: project skeleton"
```

Use **branches and many small commits** (the rubric asks for clean commits). A sensible history:

```bash
git checkout -b feature/data-pipeline   # src/data, src/features, dvc.yaml, params.yaml   -> merge
git checkout -b feature/training        # src/models, MLproject                              -> merge
git checkout -b feature/serving         # src/serving, tests/test_api.py                     -> merge
git checkout -b feature/frontend        # frontend/                                          -> merge
git checkout -b feature/monitoring      # src/monitoring, src/retrain, docker/               -> merge
git checkout -b docs                    # docs/, README, report                              -> merge
```
Commit after each logical piece (`feat:`, `test:`, `docs:`, `ci:` prefixes).

**Screenshot S01** - terminal: `git log --oneline --graph --all` (shows branches + commits).
Lecture: `Lecture 2 -- Git.pdf`. Rubric: Source control.

### DVC remote (data versioning, Assignment 1 style)
```bash
mkdir -p ../quizpulse-dvc-storage
dvc remote add -d localstorage ../quizpulse-dvc-storage
```
Lecture: `Lecture 2 -- DVC.pdf`; large files alternative: `Lecture 2 -- TrackingHugeFilesWithGitLFS.pdf`.
(Guidelines forbid cloud platforms - a local/SSH remote is fine.)

---------------------------------------------------------------------------------------------------

## 2. Data + feature pipeline with DVC and Ray (Day 1-2)

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000     # start MLflow first, see step 3, OR leave unset for local sqlite
dvc repro                       # generate -> ingest -> batches -> features(Ray) -> train -> test
dvc dag                         # the pipeline "CI DAG"
dvc metrics show                # data quality, feature throughput, model metrics
```
What each stage does: `docs/HLD.md` section 4. Change a value in `params.yaml` (e.g. `generator.n_students`) and run
`dvc repro` again - only the dependent stages re-run. That is the "CI with DVC" story.

| Screenshot | What | Rubric |
|---|---|---|
| **S02** | `dvc dag` output | CI with DVC (DAG) |
| **S03** | `dvc repro` finishing (all stages) | CI with DVC |
| **S04** | `dvc metrics show` | CI / Experiment tracking |
| **S05** | `cat reports/data_quality.json` and `reports/feature_stats.json` (shows `"engine": "ray"` + rows/s) | Data engineering throughput |
| **S06** | `dvc status` / `git status` after change (proves versioning) and, after a data change, `git checkout <old> && dvc checkout` (rollback) | Source/Data versioning |

Using the real ASSISTments data instead of the generator (optional, recommended for the report):
download "skill_builder_data" (ASSISTments 2009-2010, check licence), put it in `data/external/`, set
`data.source: assistments` in `params.yaml`, change the `ingest` stage deps to the CSV and `dvc repro`.
The synthetic generator is still used to inject drift.

Ray note: at 50k rows Ray is *slower* than pandas (start-up cost; see `feature_stats.json`: ~3.5k vs ~28k rows/s).
Say this honestly in the viva - Ray pays off on multi-million-row data; the stage keeps `engine: ray|pandas` switchable and
`tests/test_features.py::test_ray_matches_pandas` proves identical output.
Lectures: `Lecture 8 -- Ray.pdf`, `Lecture 8 -- Ray - Module 2.pdf`, `Lecture 8 -- Ray-introduction.ipynb`,
`Lecture 8 -- Ray_Distributed_Exercise.ipynb`; Spark alternative: `Lecture 7 -- Apache Spark ETL.pdf`.

---------------------------------------------------------------------------------------------------

## 3. Experiment tracking and registry with MLflow (Day 2-3)

```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db \
              --artifacts-destination ./mlartifacts --serve-artifacts     # keep running (or use docker compose)
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m src.models.train        # baseline | logreg | lightgbm  -> 3 runs
python -m src.models.register     # best model -> registry, alias "production" (fails if it doesn't beat baseline)
```
Open <http://localhost:5000>.

| Screenshot | What | Rubric |
|---|---|---|
| **S07** | Experiment `quizpulse-training`: select the 3 runs -> **Compare** (metrics table + parallel coords) | Experiment tracking |
| **S08** | One run page: Parameters, Metrics, Tags (`git_commit`, `data_version`), Artifacts (`plots/roc_curve.png`, `feature_importance.json`, `reference/`, `config/params.yaml`, `model/`) | "metrics, parameters & artifacts" |
| **S09** | **Models** tab -> `quizpulse-kt` -> version 1, alias `production`, tags | Registry |
| **S10** | Run `mlflow run . -P models=baseline,logreg` (MLproject) and the new run in the UI | MLprojects |

Why "besides autolog": every run logs git commit, data hash, validation *and* test metrics, ROC + importance plots, the
drift reference sample and `params.yaml` by hand (see `train_one()` in `src/models/train.py`).
Reproducibility rule from the guidelines: *every experiment reproducible via a Git commit hash + MLflow run ID* ->
tags `git_commit` and the run id on every model version.

Lectures: `Lecture 3 -- Tracking via MLflow.pdf`, `Lecture 3 -- MLflow_Deep_Dive.pdf`; exercises in the course repo:
`MLflow/Exercise1_Instrument_and_Compare_Runs.ipynb` (runs), `Exercise2_Package_and_Run_MLflow_Project.ipynb` (MLproject),
`Exercise3_Log_Model_with_Signature.ipynb` (signature - see `infer_signature` in `train.py`),
`Exercise4_Register_and_Promote_Model.ipynb` (registry/alias - `src/models/register.py`),
`Exercise5_Serve_Locally_and_Query.ipynb` (serving - step 6 below).

---------------------------------------------------------------------------------------------------

## 4. The API (Day 3-4)

```bash
export DATABASE_URL=sqlite:///quizpulse.db            # (Postgres is used in Docker)
python scripts/seed_history.py                        # loads batch_0 as students' history
uvicorn src.serving.app:app --port 8000
```
Open <http://localhost:8000/docs>.

| Screenshot | What | Rubric |
|---|---|---|
| **S11** | Swagger UI listing all endpoints | LLD / FastAPI |
| **S12** | `curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"student_id":"S0001","question_id":"FRA_03","skill":"Fractions"}'` and the JSON reply | API |
| **S13** | `curl localhost:8000/students/S0001/weak-topics` and `curl -X POST localhost:8000/next-questions -H 'content-type: application/json' -d '{"student_id":"S0001","n":5}'` | API |
| **S14** | `curl localhost:8000/metrics \| grep qp_` | Prometheus instrumentation |

Lectures: `Lecture 10 -- REST_API_Serving.pdf` + `Lecture 10 -- REST_API_Serving_Exercise.ipynb`.

---------------------------------------------------------------------------------------------------

## 5. Tests (Day 4)

```bash
ruff check src tests scripts
pytest -v                      # 32 tests: features (no leakage, offline==online), data, API, drift, retrain split
```
| Screenshot | What |
|---|---|
| **S15** | `pytest -v` output (all green) |
| **S16** | `reports/test_report.xml` (or the table in `docs/TEST_REPORT.md` after you re-run) |

Re-generate the numbers in `docs/TEST_REPORT.md` from **your** run before submitting.

---------------------------------------------------------------------------------------------------

## 6. MLflow's own model server (packaging requirement) (Day 4)

```bash
mlflow models serve -m "models:/quizpulse-kt@production" -p 5001 --env-manager local
curl -X POST localhost:5001/invocations -H 'Content-Type: application/json' -d '{"dataframe_records":[{"overall_acc":0.6,"skill_acc":0.5,"skill_attempts":3.0,"recent_acc5":0.6,"hours_since_last":5.0,"question_difficulty":0.5,"skill":"Fractions"}]}'
```
(In Docker: `docker compose --profile mlserve up mlflow-serve`.)  **S17**: the command + the JSON response, and one deliberately
wrong request showing MLflow's signature validation error (like Exercise 5).
Lecture: `Lecture 10 --  MLflow_Serving.pdf` (two spaces in the file name). We serve through FastAPI **and** expose
the same registered pyfunc through MLflow Serve; the UI only ever talks to FastAPI.

---------------------------------------------------------------------------------------------------

## 7. Docker Compose - the whole system (Day 5)

```bash
docker compose up --build        # first run: 3-6 minutes (builds the image, trains the first model)
docker compose ps
```
Services: `postgres`, `mlflow`, `bootstrap` (runs once), `api`, `monitor`, `frontend`, `prometheus`, `grafana`.

| URL | What |
|---|---|
| <http://localhost:8080> | **UI** (frontend container) |
| <http://localhost:8000/docs> | API |
| <http://localhost:5000> | MLflow |
| <http://localhost:3000> | Grafana (dashboard "QuizPulse - ML service monitoring", no login) |
| <http://localhost:9090> | Prometheus |
| <http://localhost:8100/health> | Monitor |

| Screenshot | What | Rubric |
|---|---|---|
| **S18** | `docker compose ps` - all services Up/healthy, frontend and api as separate services | Packaging |
| **S19** | `docker images` showing `quizpulse-backend` and `quizpulse-frontend` | Dockerised backend + frontend |
| **S20** | Prometheus -> Status -> Targets: `quizpulse-api`, `quizpulse-monitor` both **UP** | Prometheus |
| **S21** | Prometheus graph of `qp_rolling_auc` or `rate(qp_http_requests_total[1m])` | Prometheus |
| **S22** | Prometheus -> Alerts page (rules loaded) | Alerting |

Lectures: `Lecture 4 -- Dockerization.pdf`, `Lecture 5 -- Docker_Exercise.ipynb`, `Lecture 5 -- Dockerization -- Extras.pdf`.
If a service fails, read `docs/TROUBLESHOOTING` section at the end of this file.

---------------------------------------------------------------------------------------------------

## 8. The demo / evaluation flow (do it once for screenshots, then rehearse)

Keep the replay fast: `--student-fraction 0.4` replays 40% of the students (all their rows) in about a minute per batch.

```bash
# 1. normal traffic
python scripts/replay_batches.py --batch 1 2 3 --student-fraction 0.4
```
Watch Grafana (<http://localhost:3000>) and the UI tab **Pipeline & models**.
**S23** Grafana normal state (AUC ~0.75-0.78, PSI bars below the 0.2 line, drift = 0).

```bash
# 2. the drifted week (new syllabus topics + harder questions)
python scripts/replay_batches.py --batch 4 --student-fraction 0.4
```
Within ~10-20 s of the last rows: the monitor logs `DRIFT DETECTED`, `qp_drift_detected` becomes 1, the auto-retrain starts.
(Set `AUTO_RETRAIN: "0"` in `docker-compose.yml` to trigger it yourself with the UI button **Retrain now** - nicer for a live demo.)

| Screenshot | What | Rubric |
|---|---|---|
| **S24** | Grafana during drift: PSI lines above 0.2, rolling AUC falls, drift stat red | Monitoring in NRT |
| **S25** | `docker compose logs monitor` showing `DRIFT DETECTED` ... `retrain done: promoted` | Retraining automation |
| **S26** | Evidently report (UI -> **Drift report** button, or `reports/drift_report.html`) | Drift detection |
| **S27** | MLflow `quizpulse-retrain` experiment: challenger run with `production_holdout_auc`, `auc_gain_vs_production` | Experiment tracking |
| **S28** | MLflow Models tab: v2 now has alias `production`, tags `promotion_reason`, `promoted_at` | Registry/promotion |
| **S29** | UI **Pipeline & models**: KPI cards, flow, runs table with `drift` + `retrain` + `promotion` rows | Pipeline console |
| **S30** | `curl localhost:8000/health` -> `"model_version":"2"`, and one `/predict` reply with `"model_version":"2"` | Hot reload |
| **S31** | Grafana after the retrain: model version stat = 2, drift back to 0 | NRT visualisation |

Also take: **S32** UI **My learning** (mastery bars + weak topics), **S33** "Suggest 5 questions", **S34** "Try a question" gauge,
**S35** UI on a phone-width window (responsive), **S36** MLflow Models tab history, **S37** GitHub Actions run green (after you push).

Lectures: `Lecture 9 -- Monitoring via Prometheus.pdf` and the course examples `Monitoring/ai_app.py`,
`Monitoring/sample_prometheus.yml`, `Monitoring/sample_rules.yml` (our `src/serving/metrics.py`, `docker/prometheus/*` follow them);
assignment reference: `Mod 6_Assignments.pdf` (counters, gauges, histograms, summary, labels).

Honest results of the reference run (yours will differ slightly): baseline AUC 0.526, logistic regression 0.758,
LightGBM 0.760 (validation); live AUC ~0.78 on normal weeks, ~0.71 on the drifted week; retrained model
hold-out AUC 0.736 vs 0.727 for the old one on the drift regime. The gain is real but small - say so; the project is about the
*lifecycle*, not about squeezing AUC.

---------------------------------------------------------------------------------------------------

## 9. CI (Day 5)

* **DVC DAG = the CI pipeline the guideline asks for** (`dvc.yaml`: generate -> ingest -> batches -> features -> train, plus `test`).
  Lecture: `Lecture 11 -- DVC for Continuous Integration.pdf`.
* GitHub Actions (`.github/workflows/ci.yml`) runs lint, tests, the pipeline + the "beats baseline by 0.02" acceptance gate, and builds the images on every push.
  The guideline lists *CI/CD: DVC* and bans cloud platforms - **ask your professor whether GitHub Actions is acceptable**; if not, the DVC DAG alone satisfies the item.

Push to your own GitHub repo, then **S37** = Actions run page (green) and **S02** (`dvc dag`) again side by side.

---------------------------------------------------------------------------------------------------

## 10. Documents to finish (Day 6)

| File | Action |
|---|---|
| `docs/architecture.png` / `.svg` | Re-render with `dot -Tpng docs/architecture.dot -o docs/architecture.png` if you edit it |
| `docs/HLD.md`, `docs/LLD.md` | Read fully - you will be asked "why" |
| `docs/TEST_PLAN.md`, `docs/TEST_REPORT.md` | Paste your own `pytest` counts |
| `docs/USER_MANUAL.md` | Add your UI screenshots S32-S35 |
| `report/REPORT.md` | Short report (problem, data, models, results, MLOps design, drift experiment, limitations); export to PDF (`pandoc report/REPORT.md -o report/REPORT.pdf`) |
| `AI_DISCLOSURE.md` | Make it truthful |
| `README.md` | Check commands on a clean clone |

---------------------------------------------------------------------------------------------------

## 11. Viva (5 pts)

Rehearse `docs/VIVA_PREP.md`. Questions come from: the whole project, Assignments 1-10
(`Mod1_Assignments_v1.pdf`, `Mod 2_Assignments.pdf`, `Mod3_Assignment_Student.pdf`, `Mod 6_Assignments.pdf` ...),
MLOps tools, design choices, and defending them. Be ready to open any file and walk through it.

---------------------------------------------------------------------------------------------------

## Troubleshooting (first `docker compose up`)

I could not run Docker in the environment where this project was built (registry access blocked), so the Compose stack is the
part **most likely to need a small fix on your machine**. All the Python services were run for real as separate processes against
a real MLflow HTTP server (the same code the containers run). Most likely issues:

| Symptom | Fix |
|---|---|
| `mlflow` container exits / 403 "Invalid Host header" | the compose file already passes `--allowed-hosts "*"`; if your MLflow build rejects it, remove that flag and set `MLFLOW_SERVER_ALLOWED_HOSTS=*` instead |
| `bootstrap` fails at "database not reachable" | `docker compose down -v && docker compose up --build` (old Postgres volume) |
| `bootstrap` fails reading `data/...` | run `dvc repro` on the host first, or let bootstrap rebuild the data (it does automatically if files are missing) |
| Grafana shows "No data" | Prometheus targets must be UP (S20); wait 15 s; check time range = last 15 min |
| Port already in use | change the left side of `ports:` in `docker-compose.yml` and the URLs in `frontend/config.js` |
| UI badge says "API unreachable" | `frontend/config.js` must point to the API URL as seen **from your browser** (`http://localhost:8000`) |
| `pip install` conflict | keep `pandas<3` (Evidently 0.7 needs it) |
