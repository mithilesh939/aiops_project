"""Metrics, models, chronological splits, drift maths and the retrain split."""
import numpy as np
import pandas as pd
import pytest

from src.common.config import FEATURES
from src.models.evaluate import evaluate, time_split
from src.models.kt_wrapper import KTModel, SkillMeanBaseline
from src.monitoring.drift import compute_drift, psi
from src.retrain.retrain import split_for_retrain


def test_evaluate_perfect_and_random():
    y = np.array([0, 1] * 50)
    assert evaluate(y, y.astype(float))["auc"] == 1.0
    assert evaluate(y, np.full(100, 0.5))["auc"] == 0.5


def test_evaluate_single_class_gives_nan_auc():
    assert np.isnan(evaluate([1, 1, 1], [0.6, 0.7, 0.8])["auc"])


def test_time_split_is_chronological(small_df):
    train, valid = time_split(small_df, 0.2)
    assert train.ts.max() <= valid.ts.min()
    assert abs(len(valid) / len(small_df) - 0.2) < 0.01


def test_baseline_predicts_skill_mean_and_handles_unseen_skill():
    df = pd.DataFrame({"skill": ["a", "a", "b"], "correct": [1, 1, 0]})
    m = SkillMeanBaseline().fit(df.skill, df.correct)
    p = m.predict_proba(pd.DataFrame({"skill": ["a", "b", "zzz"]}))[:, 1]
    assert p[0] == 1.0 and p[1] == 0.0 and p[2] == pytest.approx(2 / 3)


def test_kt_wrapper_returns_probabilities():
    class Est:
        def predict_proba(self, X):
            return np.column_stack([np.full(len(X), .3), np.full(len(X), .7)])
    frame = pd.DataFrame({**{f: [0.5] for f in FEATURES}, "skill": ["x"]})
    out = KTModel(Est(), "lightgbm").predict(None, frame)
    assert out.shape == (1,) and 0 < out[0] < 1


def test_psi_is_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
    assert psi(a, b) < 0.05
    assert psi(a, rng.normal(1.5, 1, 5000)) > 0.25


def test_psi_binary_feature():
    ref = np.r_[np.zeros(900), np.ones(100)]
    assert psi(ref, np.r_[np.zeros(900), np.ones(100)]) < 0.01
    assert psi(ref, np.r_[np.zeros(500), np.ones(500)]) > 0.2


def test_compute_drift_flags_shifted_features():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({f: rng.random(2000) for f in FEATURES})
    cur = ref.copy()
    for f in ["overall_acc", "skill_acc", "recent_acc5"]:
        cur[f] = np.clip(cur[f] * 0.4, 0, 1)
    res = compute_drift(ref, cur)
    assert res["dataset_drift"] and "overall_acc" in res["drifted_features"]
    assert not compute_drift(ref, ref.sample(1500, random_state=0))["dataset_drift"]


def test_retrain_split_holdout_is_strictly_newer(small_df):
    d = small_df.copy()
    d["batch"] = 0
    train, hold = split_for_retrain(d, recent_rows=200, holdout_fraction=0.4)
    assert train.ts.max() <= hold.ts.min()
    assert len(hold) == 80
