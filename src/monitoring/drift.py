"""Drift detection: feature drift (PSI), prediction drift (PSI) and performance drop.

Reference  = feature sample logged with the production model (the data it was trained on).
Current    = features of the most recent predictions served by the API.
Decision   = dataset drift (>= drift_share of features with PSI > threshold) OR prediction
             drift OR rolling AUC dropped by more than ``auc_drop_threshold``.

Evidently produces the human-readable HTML report; PSI produces the machine decision
(simple, explainable, unit-testable).
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.common.config import FEATURES, REPORTS_DIR, load_params, settings, setup_logging
from src.common.db import log_run
from src.models.register import client, production_version
from src.monitoring import performance_monitor as pm

log = setup_logging("drift")
_ref_cache: dict = {}


# Monitored columns.  ``skill_attempts`` is a counter that grows for every student as time passes
# (natural, harmless non-stationarity), so it is monitored through its stationary form:
# ``cold_start`` = share of requests where the student has NO history on the skill
# (rises when new syllabus topics appear).
MONITORED = [f if f != "skill_attempts" else "cold_start" for f in FEATURES]


def with_monitored(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cold_start"] = (df["skill_attempts"].astype(float) == 0).astype(float)
    return df


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index (quantile bins from the reference; categories if few values)."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    levels = np.unique(ref)
    if len(levels) <= 5:                      # binary / categorical feature
        r = np.array([(ref == v).mean() for v in levels])
        c = np.array([(cur == v).mean() for v in levels])
        r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
        return float(np.sum((c - r) * np.log(c / r)))
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                       # (almost) constant reference
        edges = np.array([-np.inf, np.nanmedian(ref), np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def compute_drift(ref: pd.DataFrame, cur: pd.DataFrame, params: dict | None = None) -> dict:
    """Per-feature PSI and the dataset-level decision."""
    dp = (params or load_params())["drift"]
    ref, cur = with_monitored(ref), with_monitored(cur)
    feature_psi = {f: round(psi(ref[f], cur[f]), 4) for f in MONITORED}
    drifted = [f for f, v in feature_psi.items() if v > dp["psi_threshold"]]
    share = len(drifted) / len(MONITORED)
    out = {"feature_psi": feature_psi, "drifted_features": drifted, "drift_share": round(share, 3),
           "dataset_drift": share >= dp["drift_share_threshold"]}
    if "p_correct" in ref.columns and "p_correct" in cur.columns:
        out["prediction_psi"] = round(psi(ref["p_correct"], cur["p_correct"]), 4)
        out["prediction_drift"] = out["prediction_psi"] > dp["prediction_psi_threshold"]
    return out


def load_reference(version) -> tuple[pd.DataFrame, float | None]:
    """Reference sample + validation AUC of the production model (cached per version)."""
    key = str(version.version)
    if key not in _ref_cache:
        c = client()
        with tempfile.TemporaryDirectory() as tmp:
            path = c.download_artifacts(version.run_id, "reference/reference_features.parquet", tmp)
            ref = pd.read_parquet(path)
        run = c.get_run(version.run_id)
        auc = run.data.metrics.get("valid_auc") or run.data.metrics.get("test_auc")
        model = mlflow.pyfunc.load_model(f"models:/{settings.model_name}@{settings.model_alias}")
        ref["p_correct"] = model.predict(ref[FEATURES + ["skill"]].astype({f: float for f in FEATURES}))
        _ref_cache.clear()
        _ref_cache[key] = (ref, auc)
    return _ref_cache[key]


def evidently_html(ref: pd.DataFrame, cur: pd.DataFrame, path: Path) -> bool:
    """Write the Evidently data-drift HTML report.  Returns False if Evidently is unavailable."""
    try:
        from evidently import Dataset, Report
        from evidently.presets import DataDriftPreset
        cols = MONITORED + ["p_correct"]
        ref, cur = with_monitored(ref), with_monitored(cur)
        snap = Report([DataDriftPreset(columns=cols)]).run(
            Dataset.from_pandas(cur[cols].astype(float)), Dataset.from_pandas(ref[cols].astype(float)))
        path.parent.mkdir(parents=True, exist_ok=True)
        snap.save_html(str(path))
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("evidently report skipped: %s", exc)
        return False


def check(save_report: bool = True, log_to_db: bool = True, log_to_mlflow: bool = True) -> dict:
    """Full drift check against the production model.  Returns a JSON-serialisable dict."""
    t0 = time.time()
    p = load_params()["drift"]
    mv = production_version()
    if mv is None:
        return {"status": "skipped", "reason": "no production model", "drift_detected": False}
    cur = pm.recent_prediction_features(p["window"])
    if len(cur) < 200:
        res = {"status": "skipped", "reason": f"only {len(cur)} recent predictions (need 200)",
               "drift_detected": False, "n_current": int(len(cur))}
        if log_to_db:
            log_run("drift", "skipped", t0, rows=len(cur), details=res)
        return res
    ref, base_auc = load_reference(mv)
    res = compute_drift(ref, cur)
    rolling = pm.rolling_metrics(p["window"])
    auc_drop = None
    if rolling["auc"] is not None and rolling["n"] >= p["min_labeled_rows"] and base_auc:
        auc_drop = round(base_auc - rolling["auc"], 4)
    reasons = []
    if res["dataset_drift"]:
        reasons.append(f"feature drift: {len(res['drifted_features'])}/{len(MONITORED)} features PSI>{p['psi_threshold']}")
    if res.get("prediction_drift"):
        reasons.append(f"prediction drift PSI={res['prediction_psi']}")
    if auc_drop is not None and auc_drop > p["auc_drop_threshold"]:
        reasons.append(f"rolling AUC dropped {auc_drop:.3f} (> {p['auc_drop_threshold']})")
    res.update(status="ok", drift_detected=bool(reasons), reasons=reasons, rolling=rolling,
               baseline_auc=base_auc, auc_drop=auc_drop, n_current=int(len(cur)),
               model_version=str(mv.version), checked_at=time.time())
    html = REPORTS_DIR / "drift_report.html"
    if save_report:
        res["report"] = "drift_report.html" if evidently_html(ref, cur, html) else None
    run_id = None
    if log_to_mlflow:
        try:
            mlflow.set_tracking_uri(settings.mlflow_uri)
            mlflow.set_experiment("quizpulse-monitoring")
            with mlflow.start_run(run_name=f"drift-check-v{mv.version}") as run:
                mlflow.log_metrics({f"psi_{f}": v for f, v in res["feature_psi"].items()})
                mlflow.log_metric("drift_share", res["drift_share"])
                mlflow.log_metric("drift_detected", float(res["drift_detected"]))
                if res.get("prediction_psi") is not None:
                    mlflow.log_metric("prediction_psi", res["prediction_psi"])
                if rolling["auc"] is not None:
                    mlflow.log_metric("rolling_auc", rolling["auc"])
                if res.get("report") and html.exists():
                    mlflow.log_artifact(str(html), "evidently")
                run_id = run.info.run_id
        except Exception as exc:  # noqa: BLE001
            log.warning("could not log drift check to MLflow: %s", exc)
    if log_to_db:
        log_run("drift", "success", t0, rows=len(cur), details=res, mlflow_run_id=run_id)
    log.info("drift check: detected=%s reasons=%s", res["drift_detected"], reasons)
    return res
