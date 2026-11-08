#!/usr/bin/env bash
# Runs the WHOLE project locally without Docker and prints labelled sections (S01..S31).
# Take your own screenshot of each section / the matching browser page.
#   bash scripts/demo_run.sh          (from the project root, venv active, requirements-dev installed)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. MLFLOW_DISABLE_AGENT_HINT=1 MLFLOW_TRACKING_URI=http://localhost:5000 \
       DATABASE_URL=sqlite:///quizpulse.db MODEL_POLL_SECONDS=5 MONITOR_INTERVAL=10 AUTO_RETRAIN=1
LOGS=.demo_logs; mkdir -p "$LOGS"
hdr() { printf '\n\n=================== %s ===================\n' "$*"; }
wait_http() { for _ in $(seq 1 60); do curl -fs "$1" >/dev/null 2>&1 && return 0; sleep 2; done; echo "TIMEOUT waiting for $1"; return 1; }

hdr "S01 git log"
[ -d .git ] || { git init -q -b main && git add -A && git -c user.name=demo -c user.email=demo@example.com commit -qm "chore: project skeleton"; }
git log --oneline --graph --all | head
[ -d .dvc ] || dvc init -q

hdr "START MLflow server (open http://localhost:5000)"
rm -rf mlflow.db mlartifacts quizpulse.db
nohup mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///$PWD/mlflow.db \
  --artifacts-destination $PWD/mlartifacts --serve-artifacts --allowed-hosts "*" --cors-allowed-origins "*" > $LOGS/mlflow.log 2>&1 &
wait_http http://localhost:5000/health || exit 1

hdr "S03 dvc repro (generate -> ingest -> batches -> Ray features -> train -> test)"
rm -rf data/raw/* data/processed/* data/batches/* reports/*
dvc repro 2>&1 | grep -vE "Warn|warn|INFO mlflow|View run|View experiment|Downloading"
hdr "S02 dvc dag";            dvc dag
hdr "S04 dvc metrics show";   dvc metrics show --md 2>/dev/null | head -20 || dvc metrics show
hdr "S05 data quality + feature throughput"; cat reports/data_quality.json | head -20; cat reports/feature_stats.json
hdr "S07-S10 MLflow: open http://localhost:5000 -> quizpulse-training -> compare 3 runs; Models tab"
python -m src.models.register 2>&1 | grep -E "register|version"
mlflow run . -P models=baseline,logreg --env-manager=local --experiment-name quizpulse-training 2>&1 | grep -E "AUC|succeeded"

hdr "S15 pytest -v"; python -m pytest -v -p no:cacheprovider 2>&1 | tail -45

hdr "START API, monitor, UI"
python scripts/seed_history.py 2>&1 | tail -1
nohup uvicorn src.serving.app:app --port 8000 > $LOGS/api.log 2>&1 &
wait_http http://localhost:8000/health || exit 1
nohup uvicorn src.monitoring.monitor_app:app --port 8100 > $LOGS/monitor.log 2>&1 &
(cd frontend && nohup python -m http.server 8080 > ../$LOGS/ui.log 2>&1 &)
wait_http http://localhost:8100/health || exit 1
echo "UI: http://localhost:8080   API docs: http://localhost:8000/docs   MLflow: http://localhost:5000"

hdr "S12 /predict"
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"student_id":"S0001","question_id":"FRA_03","skill":"Fractions"}'; echo
hdr "S13 weak topics + next questions"
curl -s localhost:8000/students/S0001/weak-topics; echo
curl -s -X POST localhost:8000/next-questions -H 'content-type: application/json' -d '{"student_id":"S0001","n":5}'; echo
hdr "S14 Prometheus metrics of the API"; curl -s localhost:8000/metrics | grep -E "^qp_" | head -25

hdr "S17 MLflow Serve (port 5001)"
nohup mlflow models serve -m "models:/quizpulse-kt@production" -p 5001 --env-manager local > $LOGS/mlserve.log 2>&1 &
wait_http http://localhost:5001/ping && {
 curl -s -X POST localhost:5001/invocations -H 'Content-Type: application/json' -d '{"dataframe_records":[{"overall_acc":0.6,"skill_acc":0.5,"skill_attempts":3.0,"recent_acc5":0.6,"hours_since_last":5.0,"question_difficulty":0.5,"skill":"Fractions"}]}'; echo
 echo "--- malformed request (signature validation) ---"
 curl -s -X POST localhost:5001/invocations -H 'Content-Type: application/json' -d '{"dataframe_records":[{"overall_acc":0.6}]}' | head -c 300; echo; }
pkill -f "mlflow models serve"; pkill -f gunicorn

hdr "S23 NORMAL traffic: replay batches 1-3 (about 3 minutes). Screenshot the UI 'Pipeline & models' tab now"
python scripts/replay_batches.py --batch 1 2 3 --student-fraction 0.4 2>&1 | grep -E "done"
curl -s localhost:8000/monitoring/summary | python -c "import sys,json;d=json.load(sys.stdin);print('rolling',d['rolling'],'| drift?',(d['last_drift'] or {}).get('details',{}).get('drift_detected'))"
curl -s localhost:8100/metrics | grep -E "^qp_(drift_detected|rolling_auc|feature_psi)"

hdr "S24 DRIFTED week: replay batch 4 (watch UI/Grafana for DRIFT, then retrain)"
python scripts/replay_batches.py --batch 4 --student-fraction 0.4 2>&1 | grep -E "done"
sleep 45
hdr "S25 monitor log: drift -> retrain -> promoted"
grep -E "DRIFT DETECTED|auto-retrain|retrain done|drift cleared" $LOGS/monitor.log | cut -c1-220
hdr "S30 API now serves the new version"
sleep 8; curl -s localhost:8000/health; echo
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"student_id":"S0001","question_id":"FRA_03","skill":"Fractions"}' | cut -c1-120; echo
hdr "S28 model versions"
curl -s localhost:8000/model/info | python -c "import sys,json;d=json.load(sys.stdin);[print('v',v['version'],v['aliases'],v['stage'],round(v['valid_auc'] or 0,4)) for v in d['versions']]"
hdr "S29 pipeline runs"
curl -s "localhost:8000/pipeline/runs?limit=10" | python -c "import sys,json;[print(r['id'],r['kind'],r['status'],round(r['duration_s'],1),r['details'].get('reason','')[:80]) for r in json.load(sys.stdin)]"
echo
echo "DONE. Services are still running: UI http://localhost:8080 | MLflow :5000 | API docs :8000/docs | drift report: http://localhost:8000/reports/drift"
echo "Stop everything with:  pkill -f 'uvicorn src'; pkill -f 'mlflow server'; pkill -f 'http.server 8080'"
