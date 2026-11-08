"""Retraining job: check drift -> train challenger -> compare with production -> promote if better.

CLI:
    python -m src.retrain.retrain                # only retrain if drift is detected
    python -m src.retrain.retrain --force        # retrain regardless (operator action)

The comparison is fair: BOTH models are scored on the same hold-out = the most recent part of
the data (time-ordered, never seen by the challenger).  The challenger is always registered as a
new model version (audit trail); the ``production`` alias only moves if the gate passes.
"""
from __future__ import annotations

import argparse
import os
import time

import mlflow
import pandas as pd
from sqlalchemy import text

from src.common.config import FEATURES, load_params, settings, setup_logging
from src.common.db import get_engine, log_run
from src.features.build_features import build_features
from src.models import register
from src.models.evaluate import evaluate
from src.models.train import INPUT_COLS, git_commit, train_one
from src.monitoring import drift as drift_mod

log = setup_logging("retrain")


def load_attempts() -> pd.DataFrame:
    q = ("SELECT id AS order_id, user_id, skill, question_id, correct, ts, batch "
         "FROM attempts ORDER BY ts, id")
    return pd.read_sql(text(q), get_engine())


def split_for_retrain(feats: pd.DataFrame, recent_rows: int, holdout_fraction: float):
    """Older data + first part of the recent window -> train; last part of recent window -> hold-out."""
    feats = feats.sort_values(["ts", "order_id"], kind="mergesort")
    recent_rows = min(recent_rows, len(feats) // 2)
    older, recent = feats.iloc[:-recent_rows], feats.iloc[-recent_rows:]
    cut = int(len(recent) * (1 - holdout_fraction))
    train = pd.concat([older, recent.iloc[:cut]])
    return train, recent.iloc[cut:]


def retrain(force: bool = False, reason: str = "manual") -> dict:
    t0 = time.time()
    p = load_params()
    prod = register.production_version()
    if prod is None:
        raise RuntimeError("no production model to compare against - run the initial training first")

    drift_res = None
    if not force:
        drift_res = drift_mod.check(save_report=True)
        if not drift_res.get("drift_detected"):
            res = {"status": "skipped", "reason": "no drift detected", "drift": drift_res.get("reasons", [])}
            log_run("retrain", "skipped", t0, details=res)
            return res
        reason = "; ".join(drift_res["reasons"])

    attempts = load_attempts()
    min_rows = p["retrain"]["min_new_rows"]
    if len(attempts) < 2000:
        raise RuntimeError("not enough data to retrain")
    engine = os.environ.get("RETRAIN_ENGINE", "pandas")
    feats = build_features(attempts, engine=engine, params=p)
    feats[FEATURES] = feats[FEATURES].astype(float)
    train, hold = split_for_retrain(feats, p["retrain"].get("recent_rows", 6000),
                                    p["retrain"].get("holdout_fraction", 0.4))
    if len(hold) < min_rows // 2:
        raise RuntimeError(f"hold-out too small ({len(hold)})")

    # --- production model on the hold-out (same features, same rows)
    prod_model = mlflow.pyfunc.load_model(f"models:/{settings.model_name}@{settings.model_alias}")
    m_prod = evaluate(hold["correct"], prod_model.predict(hold[INPUT_COLS]))

    # --- challenger
    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(p["retrain"]["experiment"])
    tags = {"parent_production_version": str(prod.version), "trigger": reason[:200]}
    out = train_one("lightgbm", train, hold, hold, p, data_version=f"db-{len(attempts)}rows",
                    experiment=p["retrain"]["experiment"], stage="retrain", extra_tags=tags,
                    reference=feats.tail(p["drift"]["window"]))     # new "normal" = the regime just observed
    m_ch = out["valid"]
    with mlflow.start_run(run_id=out["run_id"]):
        mlflow.log_metrics({"production_holdout_auc": m_prod["auc"], "production_holdout_logloss": m_prod["logloss"],
                            "auc_gain_vs_production": m_ch["auc"] - m_prod["auc"],
                            "logloss_gain_vs_production": m_prod["logloss"] - m_ch["logloss"]})

    # --- gate: better AUC AND not worse log-loss
    delta = p["registry"]["promotion_min_auc_delta"]
    promote = (m_ch["auc"] >= m_prod["auc"] + delta) and (m_ch["logloss"] <= m_prod["logloss"] + 1e-4)
    mv = register.register_run(out["run_id"], tags={
        "valid_auc": round(m_ch["auc"], 4), "holdout_auc_production": round(m_prod["auc"], 4),
        "git_commit": git_commit(), "stage": "retrain"})
    if promote:
        register.promote(mv.version, reason=f"AUC {m_ch['auc']:.4f} vs {m_prod['auc']:.4f} (v{prod.version}); {reason}")
    else:
        register.client().set_model_version_tag(settings.model_name, mv.version, "gate", "rejected")
    res = {"status": "promoted" if promote else "rejected", "new_version": int(mv.version),
           "previous_version": int(prod.version), "reason": reason,
           "production_holdout": {k: round(v, 4) for k, v in m_prod.items()},
           "challenger_holdout": {k: round(v, 4) for k, v in m_ch.items()},
           "n_train": int(len(train)), "n_holdout": int(len(hold)), "mlflow_run_id": out["run_id"]}
    log_run("retrain", "success", t0, rows=len(train), details=res, mlflow_run_id=out["run_id"])
    log_run("promotion", "success" if promote else "skipped", time.time(), details=res, mlflow_run_id=out["run_id"])
    log.info("retrain done: %s (new v%s, AUC %.4f vs production %.4f)", res["status"], mv.version,
             m_ch["auc"], m_prod["auc"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="retrain even if no drift was detected")
    args = ap.parse_args()
    print(retrain(force=args.force))


if __name__ == "__main__":
    main()
