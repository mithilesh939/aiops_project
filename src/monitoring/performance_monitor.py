"""Online performance monitoring: join logged predictions with delayed outcomes.

``/predict`` stores every prediction (+ its features); ``/outcomes`` stores the true answer
later.  Joining the two gives the *real* production AUC, not just offline validation AUC.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from src.common.config import FEATURES
from src.common.db import get_engine
from src.models.evaluate import evaluate


def labeled_predictions(window: int | None = None) -> pd.DataFrame:
    """Predictions that already have an outcome, oldest -> newest (last ``window`` rows)."""
    q = ("SELECT p.id AS prediction_id, p.user_id, p.skill, p.question_id, p.p_correct, "
         "p.model_version, p.batch, p.ts, o.correct, o.id AS outcome_id, "
         + ", ".join(f"p.{f}" for f in FEATURES) +
         " FROM predictions p JOIN outcomes o ON o.prediction_id = p.id ORDER BY o.id DESC")
    if window:
        q += f" LIMIT {int(window)}"
    df = pd.read_sql(text(q), get_engine())
    return df.sort_values("outcome_id").reset_index(drop=True)


def recent_prediction_features(window: int) -> pd.DataFrame:
    """Feature values of the most recent predictions (with or without a label)."""
    q = ("SELECT id, skill, p_correct, model_version, batch, "
         + ", ".join(FEATURES) + f" FROM predictions ORDER BY id DESC LIMIT {int(window)}")
    return pd.read_sql(text(q), get_engine()).sort_values("id").reset_index(drop=True)


def rolling_metrics(window: int) -> dict:
    """AUC / logloss / accuracy over the last ``window`` labelled predictions."""
    df = labeled_predictions(window)
    if len(df) < 50 or df["correct"].nunique() < 2:
        return {"n": int(len(df)), "auc": None, "logloss": None, "accuracy": None}
    m = evaluate(df["correct"], df["p_correct"])
    return {"n": int(len(df)), **{k: round(v, 4) for k, v in m.items()}}


def counts() -> dict:
    eng = get_engine()
    with eng.connect() as c:
        return {
            "predictions": c.execute(text("SELECT COUNT(*) FROM predictions")).scalar() or 0,
            "outcomes": c.execute(text("SELECT COUNT(*) FROM outcomes")).scalar() or 0,
            "attempts": c.execute(text("SELECT COUNT(*) FROM attempts")).scalar() or 0,
        }
