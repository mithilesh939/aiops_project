"""Feature engineering: no leakage and offline == online parity (the most important ML test)."""
import numpy as np
import pandas as pd
import pytest

from src.common.config import FEATURES, load_params
from src.features.build_features import build_features, online_features


def test_features_have_no_nulls_and_valid_ranges(small_df):
    f = build_features(small_df, engine="pandas")
    assert f[FEATURES].isna().sum().sum() == 0
    assert f["overall_acc"].between(0, 1).all()
    assert f["skill_acc"].between(0, 1).all()
    assert (f["skill_attempts"] >= 0).all()
    assert f["question_difficulty"].between(0, 1).all()


def test_first_attempt_uses_prior_not_label(small_df):
    f = build_features(small_df, engine="pandas")
    first = f.sort_values("ts").groupby("user_id").head(1)
    assert (first["overall_acc"] == 0.5).all()          # cold start prior
    assert (first["skill_attempts"] == 0).all()


def test_no_leakage_a_label_only_affects_the_future(small_df):
    """Flipping ONE label must leave the features of that row and of every EARLIER row untouched."""
    base = build_features(small_df, engine="pandas").set_index("order_id")
    for oid in small_df.sample(5, random_state=0)["order_id"]:
        flipped = small_df.copy()
        flipped.loc[flipped.order_id == oid, "correct"] ^= 1
        after = build_features(flipped, engine="pandas").set_index("order_id")
        t = small_df.loc[small_df.order_id == oid, "ts"].iloc[0]
        upto = base.index[base["ts"] <= t]
        pd.testing.assert_frame_equal(base.loc[upto, FEATURES], after.loc[upto, FEATURES])


def test_offline_online_parity(small_df):
    """The API computes features from stored history; they must equal the offline training features."""
    params = load_params()
    offline = build_features(small_df, engine="pandas").sort_values(["ts", "order_id"]).reset_index(drop=True)
    sample = offline.sample(60, random_state=3)
    for r in sample.itertuples():
        past = small_df[(small_df.user_id == r.user_id) & (small_df.ts < r.ts)]
        q_past = small_df[(small_df.question_id == r.question_id) & (small_df.ts < r.ts)]
        feats = online_features(past, r.user_id, r.skill, r.question_id, r.ts,
                                float(q_past.correct.sum()), float(len(q_past)), params)
        for name in FEATURES:
            assert feats[name] == pytest.approx(getattr(r, name), abs=1e-9), name


def test_ray_matches_pandas(small_df):
    pytest.importorskip("ray")
    a = build_features(small_df, engine="pandas").sort_values("order_id").reset_index(drop=True)
    b = build_features(small_df, engine="ray").sort_values("order_id").reset_index(drop=True)
    np.testing.assert_allclose(a[FEATURES].to_numpy(float), b[FEATURES].to_numpy(float), atol=1e-9)
