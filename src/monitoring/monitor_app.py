"""Monitor service: Prometheus exporter + scheduled drift/performance checks + retrain trigger.

* background loop (every MONITOR_INTERVAL s): rolling AUC, PSI per feature, drift flag -> gauges
* on the first detection of drift (and AUTO_RETRAIN=1, cooldown passed) it runs the retrain job
* POST /drift/run and POST /retrain/run are the manual operator actions used by the UI
"""
from __future__ import annotations

import os
import threading
import time

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest

from src.common.config import REPORTS_DIR, load_params, setup_logging
from src.common.db import wait_for_db
from src.monitoring import drift as drift_mod
from src.monitoring import performance_monitor as pm

log = setup_logging("monitor")

G_AUC = Gauge("qp_rolling_auc", "AUC over the last N labelled predictions")
G_LOGLOSS = Gauge("qp_rolling_logloss", "Log-loss over the last N labelled predictions")
G_ACC = Gauge("qp_rolling_accuracy", "Accuracy over the last N labelled predictions")
G_N = Gauge("qp_rolling_labeled_rows", "Labelled predictions in the rolling window")
G_PSI = Gauge("qp_feature_psi", "PSI of each feature vs training reference", ["feature"])
G_PRED_PSI = Gauge("qp_prediction_psi", "PSI of predicted p_correct vs reference")
G_DRIFT = Gauge("qp_drift_detected", "1 if drift is currently detected")
G_AUC_DROP = Gauge("qp_auc_drop", "Training AUC minus rolling AUC")
C_RETRAIN = Counter("qp_retrain_runs_total", "Retrain runs", ["result"])
G_LAST_CHECK = Gauge("qp_last_drift_check_timestamp", "Unix time of last drift check")

app = FastAPI(title="QuizPulse monitor", version="1.0.0")
_state = {"last_retrain": 0.0, "drift": False, "busy": threading.Lock()}


def _update_gauges(res: dict) -> None:
    for f, v in res.get("feature_psi", {}).items():
        G_PSI.labels(f).set(v)
    if res.get("prediction_psi") is not None:
        G_PRED_PSI.set(res["prediction_psi"])
    G_DRIFT.set(1.0 if res.get("drift_detected") else 0.0)
    if res.get("auc_drop") is not None:
        G_AUC_DROP.set(res["auc_drop"])
    G_LAST_CHECK.set(time.time())


def _update_rolling() -> dict:
    r = pm.rolling_metrics(load_params()["drift"]["window"])
    G_N.set(r["n"])
    if r["auc"] is not None:
        G_AUC.set(r["auc"])
        G_LOGLOSS.set(r["logloss"])
        G_ACC.set(r["accuracy"])
    return r


def _run_retrain(force: bool, reason: str) -> dict:
    from src.retrain.retrain import retrain  # imported lazily: heavy deps
    if not _state["busy"].acquire(blocking=False):
        raise HTTPException(409, "a retrain is already running")
    try:
        res = retrain(force=force, reason=reason)
        C_RETRAIN.labels(res["status"]).inc()
        _state["last_retrain"] = time.time()
        drift_mod._ref_cache.clear()
        return res
    except Exception:
        C_RETRAIN.labels("failed").inc()
        from src.common.db import log_run
        log_run("retrain", "failed", time.time(), details={"error": "see monitor logs"})
        raise
    finally:
        _state["busy"].release()


def _loop() -> None:
    interval = float(os.environ.get("MONITOR_INTERVAL", "15"))
    auto = os.environ.get("AUTO_RETRAIN", "1") == "1"
    p = load_params()
    while True:
        try:
            _update_rolling()
            res = drift_mod.check(save_report=False, log_to_db=False, log_to_mlflow=False)
            if res.get("status") == "ok":
                _update_gauges(res)
                was, now = _state["drift"], res["drift_detected"]
                _state["drift"] = now
                if was and not now:                       # falling edge: record that the drift cleared
                    log.info("drift cleared")
                    drift_mod.check(save_report=False, log_to_mlflow=False)
                if now and not was:                       # rising edge = an event worth recording
                    log.warning("DRIFT DETECTED: %s", res["reasons"])
                    full = drift_mod.check(save_report=True)       # full report + audit row + MLflow run
                    _update_gauges(full)
                    cool = p["retrain"]["cooldown_minutes"] * 60
                    if auto and time.time() - _state["last_retrain"] > cool:
                        log.info("auto-retrain triggered")
                        try:
                            _run_retrain(force=False, reason="auto: drift detected")
                        except Exception:  # noqa: BLE001
                            log.exception("auto retrain failed")
        except Exception:  # noqa: BLE001
            log.exception("monitor loop error")
        time.sleep(interval)


@app.on_event("startup")
def startup() -> None:
    wait_for_db()
    for f in drift_mod.MONITORED:
        G_PSI.labels(f).set(0)
    G_DRIFT.set(0)
    if os.environ.get("MONITOR_LOOP", "1") == "1":
        threading.Thread(target=_loop, daemon=True, name="monitor-loop").start()


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok", "drift": _state["drift"], "retraining": _state["busy"].locked()}


@app.post("/drift/run")
def run_drift():
    res = drift_mod.check(save_report=True)
    if res.get("status") == "ok":
        _update_gauges(res)
    return res


@app.post("/retrain/run")
def run_retrain():
    return _run_retrain(force=True, reason="manual (operator)")


@app.get("/reports/drift")
def drift_report():
    path = REPORTS_DIR / "drift_report.html"
    if not path.exists():
        raise HTTPException(404, "no drift report yet - run a drift check first")
    return FileResponse(path, media_type="text/html")
