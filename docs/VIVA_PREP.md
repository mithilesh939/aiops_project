# Viva preparation (5 marks: explain project, assignments 1-10, tools, design choices, defend choices)

Practise answering **out loud, without reading**. For every answer be ready to open the file named in brackets.
The course's "whiteboard test" means: if you cannot explain it, it is assumed not to be yours.

## A. Explain the whole project (60-90 s pitch)
"QuizPulse predicts the probability that a student answers the next quiz question correctly (knowledge tracing), turns it into topic mastery and weak topics,
and recommends the next questions. The model is simple on purpose; the project is the AIOps lifecycle: DVC-versioned data pipeline with Ray, MLflow tracking and registry,
FastAPI serving in Docker Compose with a separate UI, Prometheus/Grafana monitoring, Evidently/PSI drift detection and an automatic retrain-compare-promote loop."
Draw `docs/architecture.png` from memory: data -> features -> train -> registry -> API -> Postgres -> monitor -> retrain -> registry.

## B. Design and implementation choices (and how to defend them)
1. **Why split by time, not randomly?** Random split leaks the future into training; in production we only ever predict forward. [`evaluate.time_split`, `make_batches.py`]
2. **How do you avoid feature leakage?** Every feature uses only earlier attempts (`shift(1)`, cumulative counts minus the current row). Test flips a label and checks earlier features do not move. [`test_features.py`]
3. **What is training/serving skew and how did you prevent it?** Online features call the same `user_features()` on the stored history; a test compares offline vs online on random rows. [`online_features`]
4. **Why LightGBM when logistic regression is as good?** Spec asked for a tree model as the main one; on this (mostly linear, synthetic) data they tie (0.760 vs 0.758). Honest answer: the model matters less than the pipeline; LightGBM can capture interactions on real data.
5. **Why is the baseline AUC ~0.53?** It ignores the student, so it can barely rank rows within a skill - this shows personalisation is what adds value.
6. **Why Ray, and why is it slower here?** Rubric asks for Ray/Spark/K8s; students are independent so `groupby(user).map_groups` parallelises. At 50k rows start-up overhead dominates (3.5k vs 28k rows/s); benefits appear at millions of rows. Engine is switchable and tested equal.
7. **Why PSI for drift and what threshold?** PSI < 0.1 stable, 0.1-0.2 moderate, > 0.2 significant (industry rule of thumb). Dataset drift = >= 30% of features above 0.2, or prediction drift, or live AUC drop > 0.04. Evidently gives the human-readable report.
8. **Why did you replace `skill_attempts` by `cold_start` in drift monitoring?** A counter grows for every student over time, so PSI flagged "drift" on perfectly normal weeks (a real false alarm I hit). `cold_start` (no history on this skill) is stationary and rises when a new syllabus topic appears.
9. **Why is the drift reference "recent part of training data"?** Early rows have immature histories (many cold starts), which would again look like drift. After a retrain the reference becomes the regime that triggered it, otherwise the drift flag would never clear.
10. **How does the retrain gate work and why?** Same time-ordered hold-out for both models, never used for training; promote only if AUC >= old + 0.002 and log-loss not worse. Rejected models still get registered for audit. Avoids replacing a model by noise.
11. **How does the API get the new model without restart?** Alias `production` in the registry; a background thread polls it every few seconds and swaps the pyfunc model under a lock. Rollback = move the alias back. [`model_loader.py`]
12. **Why aliases and not stages?** Stages are deprecated in newer MLflow; aliases are flexible (`production`, `candidate`). (Lecture 3 / Exercise 4 shows both.)
13. **Where is delayed ground truth handled?** `/predict` stores the prediction + features; `/outcomes` stores the label later; joining them gives live AUC. [`performance_monitor.py`]
14. **Why Postgres?** Online features need history, monitoring needs predictions + outcomes, the audit trail needs durable storage; multiple containers share it. SQLite is used in tests.
15. **Loose coupling UI <-> backend?** The UI is a separate nginx container; only fetch() calls to the REST URL in `config.js`; CORS on the API. Either side can be replaced.
16. **What do you monitor and why?** Rate, latency (histogram), errors, prediction distribution, model version (service health) + rolling AUC, PSI, drift flag (model health). Labels instead of hierarchies (Lecture 9). Alert rules for API down, latency, drift, AUC low, error rate.
17. **Counter vs gauge vs histogram?** Counter only goes up (requests); gauge goes up/down (rolling AUC, model version); histogram buckets observations to compute quantiles (latency, p_correct). [`metrics.py`]
18. **What does `dvc repro` do and how is it CI?** Builds the DAG from `dvc.yaml`, hashes deps/params/outs in `dvc.lock` and re-runs only stages whose inputs changed -> reproducible, cached pipeline (Lecture 11).
19. **Git vs DVC vs MLflow - who versions what?** Git: code + small files + `dvc.lock`; DVC: data/feature files; MLflow: runs, params, metrics, models. Reproduce = git commit + data version + MLflow run id.
20. **MLproject - what does it give you?** A declared entry point with parameters and an environment (`python_env.yaml`) so `mlflow run .` reproduces training the same way anywhere.
21. **Docker: why one backend image for four services?** Same code and libraries everywhere = environment parity; different `command` per service. Frontend is its own image.
22. **Compose: what problem does it solve?** Defines and starts a multi-container system (networks, volumes, ordering via healthchecks / `depends_on`) in one command. `bootstrap` runs once and the api waits for it to complete.
23. **Security?** Parameterised SQL (SQLAlchemy), validated inputs (pydantic), no secrets in code, local only. Not done: TLS, auth, encryption at rest (guideline mentions encryption; say it is future work).
24. **Scalability?** API is stateless -> more workers/containers; Ray for features; Postgres indexes on (user, ts) and (question, ts). Limits: feature build is per-request Python (3.5 ms).
25. **Explainability?** Feature importance (LightGBM) / coefficients (LR) logged as artifacts; per-prediction features are returned by `/predict`.
26. **Weak point you know about?** Retrained gain is small (+0.009 AUC) and synthetic data is mostly linear; drift is injected, not natural; Docker stack must be re-verified on the lab machine; no auth.

## C. Questions from the assignments (map to files in the course repo)
| Assignment | Typical question | Where to revise |
|---|---|---|
| Mod 1 (experiments, DVC, reproducibility) | six hidden technical-debt categories; log params/metrics; rollback with `git checkout` + `dvc checkout`; reproduce a partner's run from a commit | `Mod1_Assignments_v1.pdf`, `Mod1_Course_Content.pdf`, Lecture 2/3 |
| Mod 2 (Spark vs Ray) | difference between Spark data parallelism and Ray task parallelism; lazy evaluation; shuffle; benchmark | `Mod 2_Assignments.pdf`, Lecture 7/8 |
| Mod 3 (Docker, K8s) | multi-stage builds and image size; compose with cache; Job parallelism/pod placement | `Mod3_Assignment_Student.pdf`, Lecture 4-6 |
| Mod 4-6 (serving, monitoring) | REST vs MLflow vs Ray Serve; Prometheus metric types; alert rules | `Mod4_Course_Content.pdf`, `Mod 6_Assignments.pdf`, Lecture 9/10 |
(Check which module covers assignments 4-10 in your course; the PDFs list them.)

## D. If they challenge you
* "Why not Kafka/Kubernetes?" - out of scope for 4 weeks; replay script plays the role of the stream, Compose of the orchestrator; the API is stateless so it would port.
* "Isn't synthetic data fake?" - used only to inject *controlled* drift and mimic Sanrathi's schema; the pipeline also supports ASSISTments (`data.source`). Metrics are from real runs of that data.
* "Show me it works" - run `scripts/replay_batches.py --batch 4 --student-fraction 0.4` and watch Grafana/UI.
