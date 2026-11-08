"""Split the attempts table into time-ordered batches (simulated "weeks").

batch_0            -> initial training data (weeks 1..train_weeks)
batch_1..batch_N   -> live batches replayed to the API; the last one holds the drift

Splitting is strictly by time (never random) to avoid leakage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import BATCH_DIR, PROCESSED_DIR, REPORTS_DIR, load_params, setup_logging

log = setup_logging("batches")


def assign_batches(df: pd.DataFrame, train_weeks: int, n_live: int) -> pd.DataFrame:
    """Add a ``batch`` column: 0 for training weeks, 1..n_live for the remaining weeks."""
    df = df.copy()
    weeks = sorted(df["week"].unique())
    live_weeks = [w for w in weeks if w > train_weeks]
    batch = np.zeros(len(df), dtype=int)
    if live_weeks:
        # spread live weeks over n_live batches, keeping order (last batch = last weeks)
        chunks = np.array_split(np.array(live_weeks), min(n_live, len(live_weeks)))
        for i, chunk in enumerate(chunks, start=1):
            batch[df["week"].isin(chunk).to_numpy()] = i
    df["batch"] = batch
    return df


def main() -> None:
    p = load_params()["batches"]
    df = pd.read_parquet(PROCESSED_DIR / "attempts.parquet")
    df = assign_batches(df, p["train_weeks"], p["n_live_batches"])
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for old in BATCH_DIR.glob("batch_*.parquet"):
        old.unlink()
    summary = {}
    for b, part in df.groupby("batch"):
        path = BATCH_DIR / f"batch_{b}.parquet"
        part.to_parquet(path, index=False)
        summary[f"batch_{b}"] = {
            "rows": int(len(part)),
            "weeks": sorted(int(w) for w in part["week"].unique()),
            "positive_rate": round(float(part["correct"].mean()), 4),
            "new_skills_vs_batch0": [],
        }
    base_skills = set(df[df.batch == 0]["skill"])
    for b, part in df.groupby("batch"):
        summary[f"batch_{b}"]["new_skills_vs_batch0"] = sorted(set(part["skill"]) - base_skills)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(REPORTS_DIR / "batches.json").write_text(json.dumps(summary, indent=2))
    log.info("wrote %d batches: %s", len(summary), {k: v["rows"] for k, v in summary.items()})


if __name__ == "__main__":
    main()
