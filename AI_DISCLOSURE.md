# AI use disclosure

(Required by the course "Fair and Responsible Use of AI" code of conduct. **Edit this so it is accurate for you.**)

1. **Tools used:** Claude (Anthropic) via a coding-agent session.
2. **How they were used:** Claude generated the initial implementation of this project from the project specification and the course's AI Application
   Development/Evaluation guidelines: the data and feature pipeline, MLflow training/registry code, the FastAPI service, the web UI, the monitoring/drift/retrain
   modules, Docker/Compose/Prometheus/Grafana configuration, tests, CI workflow and the documents in `docs/`. Claude also ran the pipeline, tests and the
   drift -> retrain loop in its sandbox to check them and fixed bugs it found (for example a false drift alarm caused by a growing counter feature).
3. **Impact:** AI wrote most of the first version of the code and documents. **[Student: describe what *you* did - which files you read and understood, what you changed
   or decided (design choices, thresholds, dataset choice), what you debugged, and the experiments/screenshots you produced yourself on your machine.]**
4. **What was NOT AI-generated:** all screenshots, logs and metrics submitted must come from your own runs; none were fabricated. The numbers quoted in `docs/TEST_REPORT.md` and
   `report/REPORT.md` are from a real run and should be replaced by your own.
5. **Not verified by the AI:** the Docker Compose stack, Prometheus/Grafana UIs, browser rendering of the UI and GitHub Actions could not be run in its sandbox - tested/fixed by the student.
