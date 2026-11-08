"""MLflow pyfunc wrapper so every model (baseline / LR / LightGBM) exposes the SAME contract.

Input  : DataFrame with the six numeric features + ``skill`` (string)
Output : numpy array of P(correct) in [0, 1]

Because the wrapper is a pyfunc, the registered model can be served by FastAPI
(``mlflow.pyfunc.load_model``) *or* directly by ``mlflow models serve``.
"""
from __future__ import annotations

import mlflow.pyfunc
import numpy as np
import pandas as pd

FEATURES = ["overall_acc", "skill_acc", "skill_attempts", "recent_acc5",
            "hours_since_last", "question_difficulty"]


class SkillMeanBaseline:
    """Baseline: predict each skill's average correctness (global mean for unseen skills)."""

    def __init__(self):
        self.skill_mean: dict[str, float] = {}
        self.global_mean: float = 0.5

    def fit(self, skills: pd.Series, y: pd.Series) -> "SkillMeanBaseline":
        self.global_mean = float(y.mean())
        self.skill_mean = y.groupby(skills.values).mean().to_dict()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = df["skill"].map(self.skill_mean).fillna(self.global_mean).to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


class KTModel(mlflow.pyfunc.PythonModel):
    """Wraps any estimator exposing ``predict_proba``."""

    def __init__(self, estimator, kind: str):
        self.estimator = estimator
        self.kind = kind

    def predict(self, context, model_input, params=None):  # noqa: D401
        df = pd.DataFrame(model_input)
        if self.kind == "baseline":
            proba = self.estimator.predict_proba(df)
        else:
            proba = self.estimator.predict_proba(df[FEATURES].astype(float))
        return np.clip(proba[:, 1], 1e-6, 1 - 1e-6)
