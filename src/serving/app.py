"""QuizPulse REST API (FastAPI).  The ONLY interface between the UI and the model.

Endpoints
---------
POST /predict                    p_correct for (student, question, skill)
POST /outcomes                   record the real answer (delayed label)
GET  /students/{id}/weak-topics  topics ranked by lowest mastery
GET  /students/{id}/mastery      same data, all topics (alias)
POST /next-questions             adaptive question recommendations
GET  /students, /skills          lookups for the UI
GET  /model/info                 registry versions + currently served version
GET  /pipeline/runs|status       audit trail of pipeline executions
GET  /monitoring/summary         live rolling AUC, drift status, counts
POST /admin/reload|drift-check|retrain   operator actions
GET  /health, /metrics           liveness and Prometheus exposition
"""
from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

from src.common.config import FEATURES, REPORTS_DIR, load_params, settings, setup_logging
from src.common.db import Attempt, Outcome, PipelineRun, Prediction, get_engine, session_scope, wait_for_db
from src.features.build_features import online_features
from src.monitoring import performance_monitor as pm
from src.serving import mastery, metrics
from src.serving.model_loader import holder
from src.serving.schemas import (
    Features,
    HealthResponse,
    NextQuestionsRequest,
    NextQuestionsResponse,
    OutcomeRequest,
    OutcomeResponse,
    PredictRequest,
    PredictResponse,
    WeakTopicsResponse,
)

log = setup_logging("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start-up: connect DB, load the production model, start the registry watcher."""
    wait_for_db()
    metrics.BUILD_INFO.info({"version": "1.0.0", "service": "quizpulse-api"})
    if os.environ.get("QP_LOAD_MODEL", "1") == "1":
        try:
            holder.load_from_registry()
        except Exception as exc:  # noqa: BLE001
            log.warning("no production model yet (%s); will keep polling", exc)
        holder.watch(float(os.environ.get("MODEL_POLL_SECONDS", "10")))
    yield


app = FastAPI(title="QuizPulse API", version="1.0.0", lifespan=lifespan,
              description="Knowledge-tracing service: predicts P(correct), mastery and next questions.")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def instrument(request: Request, call_next):
    """Count + time every request and log it (labels use the route template, not the raw URL)."""
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception:  # noqa: BLE001
        log.exception("unhandled error on %s", request.url.path)
        raise
    finally:
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        dur = time.perf_counter() - start
        metrics.HTTP_REQUESTS.labels(request.method, endpoint, str(status)).inc()
        metrics.HTTP_LATENCY.labels(endpoint).observe(dur)
        if endpoint not in ("/metrics", "/health"):
            log.info("%s %s -> %s (%.1f ms)", request.method, request.url.path, status, dur * 1000)


def _require_model() -> None:
    if not holder.loaded:
        metrics.ERRORS.labels("model", "not_loaded").inc()
        raise HTTPException(503, "model not loaded yet - registry has no 'production' version")


# ----------------------------------------------------------------------------- core
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Predict the probability that the student answers the question correctly."""
    _require_model()
    ts = req.timestamp if req.timestamp is not None else mastery.platform_now()
    try:
        with metrics.FEATURE_LATENCY.time():
            hist = mastery.student_history(req.student_id, before=ts)
            q_sum, q_n = mastery.question_stats(req.question_id, ts)
            feats = online_features(hist, req.student_id, req.skill, req.question_id, ts,
                                    q_sum, q_n, load_params())
        frame = pd.DataFrame([{**feats, "skill": req.skill}])
        p = float(holder.predict(frame)[0])
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        metrics.ERRORS.labels("/predict", type(exc).__name__).inc()
        log.exception("prediction failed")
        raise HTTPException(500, f"prediction failed: {exc}") from exc
    with session_scope() as s:
        row = Prediction(user_id=req.student_id, skill=req.skill, question_id=req.question_id,
                         p_correct=p, model_version=holder.version, ts=ts, batch=req.batch, **feats)
        s.add(row)
        s.flush()
        pid = row.id
    metrics.PREDICTIONS.labels(holder.version).inc()
    metrics.PREDICTION_VALUE.observe(p)
    return PredictResponse(prediction_id=pid, student_id=req.student_id, question_id=req.question_id,
                           skill=req.skill, p_correct=round(p, 6), model_version=holder.version,
                           features=Features(**feats))


@app.post("/outcomes", response_model=OutcomeResponse)
def outcomes(req: OutcomeRequest) -> OutcomeResponse:
    """Record what the student actually answered (the delayed label)."""
    with session_scope() as s:
        pred = s.get(Prediction, req.prediction_id) if req.prediction_id else None
        if req.prediction_id and pred is None:
            raise HTTPException(404, f"prediction {req.prediction_id} not found")
        student = pred.user_id if pred else req.student_id
        question = pred.question_id if pred else req.question_id
        skill = pred.skill if pred else req.skill
        if not (student and question and skill):
            raise HTTPException(422, "give prediction_id OR student_id + question_id + skill")
        ts = pred.ts if pred else (req.timestamp or time.time())
        batch = pred.batch if pred and pred.batch is not None else 0
        att = Attempt(user_id=student, skill=skill, question_id=question, correct=req.correct,
                      ts=ts, batch=batch)
        s.add(att)
        s.add(Outcome(prediction_id=req.prediction_id, user_id=student, correct=req.correct,
                      ts=time.time()))
        s.flush()
        aid = att.id
    metrics.OUTCOMES.labels(str(req.correct)).inc()
    mastery._catalog_cache["df"] = None
    return OutcomeResponse(status="recorded", attempt_id=aid)


@app.get("/students/{student_id}/weak-topics", response_model=WeakTopicsResponse)
def weak_topics(student_id: str, include_strong: bool = False) -> WeakTopicsResponse:
    """Topics ranked by lowest mastery; by default only the weak ones (< 0.5)."""
    _require_model()
    topics = mastery.student_mastery(student_id)
    if not topics:
        raise HTTPException(404, "no question catalog yet")
    if not include_strong:
        topics = [t for t in topics if t["weak"]]
    return WeakTopicsResponse(student_id=student_id, threshold=mastery.WEAK_THRESHOLD,
                              model_version=holder.version, topics=topics)


@app.get("/students/{student_id}/mastery", response_model=WeakTopicsResponse)
def student_mastery(student_id: str) -> WeakTopicsResponse:
    """Mastery of every topic (weakest first)."""
    return weak_topics(student_id, include_strong=True)


@app.post("/next-questions", response_model=NextQuestionsResponse)
def next_questions(req: NextQuestionsRequest) -> NextQuestionsResponse:
    """Recommend the next questions, targeting the student's weakest topics."""
    _require_model()
    qs = mastery.next_questions(req.student_id, req.n)
    return NextQuestionsResponse(student_id=req.student_id, model_version=holder.version, questions=qs)


# ----------------------------------------------------------------------------- lookups
@app.get("/students")
def students(q: str = "", limit: int = 20):
    """Student ids (with number of attempts) for the UI picker."""
    sql = ("SELECT user_id, COUNT(*) AS attempts FROM attempts WHERE user_id LIKE :q "
           "GROUP BY user_id ORDER BY attempts DESC LIMIT :n")
    with get_engine().connect() as c:
        rows = c.execute(text(sql), {"q": f"{q}%", "n": min(limit, 200)}).fetchall()
    return [{"student_id": r[0], "attempts": r[1]} for r in rows]


@app.get("/skills")
def skills():
    cat = mastery.question_catalog()
    return sorted(cat["skill"].unique().tolist()) if len(cat) else []


@app.get("/questions")
def questions(skill: str):
    cat = mastery.question_catalog()
    return sorted(cat[cat["skill"] == skill]["question_id"].tolist())


# ----------------------------------------------------------------------------- ops
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    db_ok = True
    try:
        get_engine().connect().close()
    except Exception:  # noqa: BLE001
        db_ok = False
    ok = db_ok and holder.loaded
    return HealthResponse(status="ok" if ok else "degraded", model_loaded=holder.loaded,
                          model_version=holder.version, database=db_ok)


@app.get("/metrics")
def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info")
def model_info():
    """Registered versions with their key tags + which one is served."""
    try:
        c = holder_client()
        versions = []
        for v in c.search_model_versions(f"name='{settings.model_name}'"):
            full = c.get_model_version(settings.model_name, v.version)
            run = c.get_run(v.run_id) if v.run_id else None
            versions.append({
                "version": int(v.version), "aliases": list(full.aliases),
                "created": v.creation_timestamp / 1000, "run_id": v.run_id,
                "model_kind": (run.data.tags.get("model_kind") if run else None),
                "stage": (run.data.tags.get("stage") if run else None),
                "valid_auc": (run.data.metrics.get("valid_auc") if run else None),
                "git_commit": (run.data.tags.get("git_commit") if run else None),
                "tags": dict(full.tags)})
        versions.sort(key=lambda d: d["version"], reverse=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"registry unavailable: {exc}") from exc
    return {"model_name": settings.model_name, "serving_version": holder.version,
            "loaded_at": holder.loaded_at, "versions": versions}


def holder_client():
    import mlflow
    mlflow.set_tracking_uri(settings.mlflow_uri)
    return mlflow.MlflowClient()


@app.get("/pipeline/runs")
def pipeline_runs(limit: int = 30, kind: str | None = None):
    """Audit trail of pipeline executions (ingest, train, drift, retrain, promotion)."""
    with session_scope() as s:
        q = s.query(PipelineRun).order_by(PipelineRun.id.desc())
        if kind:
            q = q.filter(PipelineRun.kind == kind)
        rows = q.limit(min(limit, 200)).all()
        return [{"id": r.id, "kind": r.kind, "status": r.status, "started": r.started,
                 "duration_s": r.duration_s, "rows": r.rows, "mlflow_run_id": r.mlflow_run_id,
                 "details": json.loads(r.details or "{}")} for r in rows]


@app.get("/pipeline/status")
def pipeline_status():
    """Counts of successful / failed runs per stage (the 'stock-taking' console)."""
    with get_engine().connect() as c:
        rows = c.execute(text("SELECT kind, status, COUNT(*), AVG(duration_s), MAX(finished) "
                              "FROM pipeline_runs GROUP BY kind, status")).fetchall()
    out: dict = {}
    for kind, status, n, avg, last in rows:
        out.setdefault(kind, {})[status] = {"count": n, "avg_duration_s": round(avg or 0, 2), "last": last}
    return out


@app.get("/pipeline/throughput")
def pipeline_throughput():
    """Speed / throughput of the batch pipeline stages (written by DVC stages to reports/*.json)."""
    out = {}
    for name in ("data_quality", "feature_stats"):
        path = REPORTS_DIR / f"{name}.json"
        if path.exists():
            out[name] = json.loads(path.read_text())
    return out


@app.get("/monitoring/summary")
def monitoring_summary():
    """Everything the UI needs for the live health card."""
    p = load_params()["drift"]
    last_drift = pipeline_runs(limit=1, kind="drift")
    return {"counts": pm.counts(), "rolling": pm.rolling_metrics(p["window"]),
            "last_drift": last_drift[0] if last_drift else None,
            "serving_version": holder.version}


@app.get("/reports/drift")
def drift_report():
    """The latest Evidently HTML drift report (written by the monitor service)."""
    path = REPORTS_DIR / "drift_report.html"
    if not path.exists():
        raise HTTPException(404, "no drift report yet - run a drift check first")
    return FileResponse(path, media_type="text/html")


def _monitor_call(path: str):
    try:
        r = requests.post(f"{settings.monitor_url}{path}", timeout=600)
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"monitor service error: {exc}") from exc


@app.post("/admin/reload")
def admin_reload():
    try:
        changed = holder.load_from_registry()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, str(exc)) from exc
    return {"reloaded": changed, "serving_version": holder.version}


@app.post("/admin/drift-check")
def admin_drift_check():
    return _monitor_call("/drift/run")


@app.post("/admin/retrain")
def admin_retrain():
    out = _monitor_call("/retrain/run")
    try:
        holder.load_from_registry()
    except Exception:  # noqa: BLE001
        pass
    return out


assert FEATURES
