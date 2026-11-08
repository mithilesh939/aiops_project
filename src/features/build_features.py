"""Feature engineering for knowledge tracing (offline AND online, same code).

All features are computed from *past* attempts only (no leakage):

overall_acc          student's running accuracy over all previous attempts
skill_acc            student's running accuracy on this skill
skill_attempts       number of previous attempts by the student on this skill
recent_acc5          accuracy over the student's last 5 attempts
hours_since_last     hours since the student's previous attempt (capped)
question_difficulty  smoothed historical correct rate of the question (all students)

Offline: ``build_features`` runs per-student feature code in parallel with Ray Data
(or plain pandas).  Online: ``online_features`` re-uses the very same per-student
function on the stored history, which guarantees train/serve parity (see tests).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import BATCH_DIR, FEATURES, PROCESSED_DIR, REPORTS_DIR, load_params, setup_logging

log = setup_logging("features")
PRIOR = 0.5   # prior correct-rate used for cold starts


def user_features(g: pd.DataFrame, window: int = 5, time_cap_hours: float = 720.0) -> pd.DataFrame:
    """Per-student features. ``g`` holds all attempts of ONE student."""
    g = g.sort_values(["ts", "order_id"], kind="mergesort").copy()
    c = g["correct"].astype(float)
    n_prev = np.arange(len(g), dtype=float)
    overall = c.cumsum().shift(1) / np.where(n_prev == 0, np.nan, n_prev)
    by_skill = g.groupby("skill")["correct"]
    s_n = by_skill.cumcount().astype(float)
    s_sum = by_skill.cumsum().astype(float) - c
    skill_acc = s_sum / s_n.replace(0.0, np.nan)
    recent = c.shift(1).rolling(window, min_periods=1).mean()
    hours = (g["ts"].diff() / 3600.0).clip(upper=time_cap_hours).fillna(time_cap_hours)

    overall = overall.fillna(PRIOR)
    g["overall_acc"] = overall
    g["skill_acc"] = skill_acc.fillna(overall)
    g["skill_attempts"] = s_n
    g["recent_acc5"] = recent.fillna(overall)
    g["hours_since_last"] = hours
    return g


def add_question_difficulty(df: pd.DataFrame, prior_weight: float) -> pd.DataFrame:
    """Smoothed past correct-rate of each question across ALL students (time ordered)."""
    df = df.sort_values(["ts", "order_id"], kind="mergesort").copy()
    grp = df.groupby("question_id")["correct"]
    q_n = grp.cumcount().astype(float)
    q_sum = grp.cumsum().astype(float) - df["correct"].astype(float)
    df["question_difficulty"] = (q_sum + PRIOR * prior_weight) / (q_n + prior_weight)
    return df


def build_features(df: pd.DataFrame, engine: str = "pandas", params: dict | None = None) -> pd.DataFrame:
    """Compute the six features for every row of ``df`` (sorted by time on return)."""
    fp = (params or load_params())["features"]
    window, cap = fp["recent_window"], float(fp["time_cap_hours"])
    df = df.copy()
    if engine == "ray":
        import ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR",
                     num_cpus=None)
        ds = ray.data.from_pandas(df)
        ds = ds.groupby("user_id").map_groups(
            lambda g: user_features(g, window, cap), batch_format="pandas")
        out = ds.to_pandas()
    else:
        out = pd.concat([user_features(g, window, cap) for _, g in df.groupby("user_id")],
                        ignore_index=True)
    out = out.astype({f: "float64" for f in FEATURES if f in out.columns})  # Ray returns arrow dtypes
    out = add_question_difficulty(out, fp["difficulty_prior_weight"])
    return out.sort_values(["ts", "order_id"], kind="mergesort").reset_index(drop=True)


def online_features(history: pd.DataFrame, user_id: str, skill: str, question_id: str,
                    ts: float, q_sum: float, q_n: float, params: dict | None = None) -> dict:
    """Features for ONE hypothetical next attempt.

    ``history``: the student's past attempts (columns: skill, correct, ts).
    ``q_sum``/``q_n``: correct count / attempts of the question before ``ts``.
    The student's per-user code is re-used on ``history + target row`` and the last row
    is returned, which makes the online features identical to the offline ones.
    """
    fp = (params or load_params())["features"]
    h = history[["skill", "correct", "ts"]].copy() if len(history) else \
        pd.DataFrame({"skill": [], "correct": [], "ts": []})
    h = h[h["ts"] < ts]
    target = pd.DataFrame({"skill": [skill], "correct": [0], "ts": [ts]})
    both = pd.concat([h, target], ignore_index=True)
    both["user_id"] = user_id
    both["order_id"] = np.arange(len(both))
    row = user_features(both, fp["recent_window"], float(fp["time_cap_hours"])).iloc[-1]
    k = fp["difficulty_prior_weight"]
    return {
        "overall_acc": float(row["overall_acc"]),
        "skill_acc": float(row["skill_acc"]),
        "skill_attempts": float(row["skill_attempts"]),
        "recent_acc5": float(row["recent_acc5"]),
        "hours_since_last": float(row["hours_since_last"]),
        "question_difficulty": float((q_sum + PRIOR * k) / (q_n + k)),
    }


def load_all_batches() -> pd.DataFrame:
    files = sorted(BATCH_DIR.glob("batch_*.parquet"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise FileNotFoundError("no batches found - run `dvc repro` first")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", default=None, help="ray | pandas (default: params.yaml)")
    args = ap.parse_args()
    p = load_params()
    engine = args.engine or p["features"]["engine"]
    df = load_all_batches()
    t0 = time.time()
    feats = build_features(df, engine=engine, params=p)
    secs = time.time() - t0
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    stats = {
        "engine": engine, "rows": int(len(feats)), "seconds": round(secs, 3),
        "throughput_rows_per_s": round(len(feats) / max(secs, 1e-6)),
        "feature_means": {f: round(float(feats[f].mean()), 4) for f in FEATURES},
        "null_counts": {f: int(feats[f].isna().sum()) for f in FEATURES},
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(REPORTS_DIR / "feature_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("features: engine=%s rows=%d in %.2fs (%d rows/s)", engine, len(feats), secs,
             stats["throughput_rows_per_s"])


if __name__ == "__main__":
    main()
