"""Data ingestion + validation stage.

Reads the raw source (synthetic generator output or the public ASSISTments
2009-10 skill-builder CSV), runs data-quality checks, cleans the table and writes
``data/processed/attempts.parquet`` plus a JSON data-quality report.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR, load_params, setup_logging
from src.data.synthetic_generator import T0, WEEK_SECONDS

log = setup_logging("ingest")
REQUIRED = ["user_id", "skill", "question_id", "correct", "ts", "week", "order_id"]


def load_assistments(path: str, n_weeks: int) -> pd.DataFrame:
    """Map the ASSISTments skill-builder CSV onto the QuizPulse schema.

    ASSISTments has no wall-clock time, so ``order_id`` (global event order) is turned
    into synthetic timestamps and split into equally sized "weeks".
    """
    raw = pd.read_csv(path, encoding="latin-1", low_memory=False)
    raw = raw.dropna(subset=["skill_name", "problem_id", "user_id", "correct"])
    df = pd.DataFrame({
        "user_id": "S" + raw["user_id"].astype(int).astype(str),
        "skill": raw["skill_name"].astype(str),
        "question_id": "P" + raw["problem_id"].astype(int).astype(str),
        "correct": raw["correct"].astype(int),
        "order_id": raw["order_id"].astype(int),
    })
    df = df.sort_values("order_id", kind="mergesort").drop_duplicates("order_id").reset_index(drop=True)
    rank = np.arange(len(df))
    df["ts"] = T0 + rank * (n_weeks * WEEK_SECONDS / len(df))
    df["week"] = np.minimum((rank * n_weeks // len(df)) + 1, n_weeks)
    return df


def validate(df: pd.DataFrame) -> dict:
    """Return a data-quality report; raise if a hard check fails."""
    report = {"rows": int(len(df)), "checks": {}}
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    report["checks"]["schema_ok"] = not missing_cols
    if missing_cols:
        raise ValueError(f"missing columns: {missing_cols}")
    report["checks"]["no_nulls"] = bool(df[REQUIRED].notna().all().all())
    report["checks"]["target_binary"] = bool(df["correct"].isin([0, 1]).all())
    report["checks"]["timestamps_sorted"] = bool(df["ts"].is_monotonic_increasing)
    report["checks"]["min_rows_ok"] = bool(len(df) >= 1000)
    report["students"] = int(df.user_id.nunique())
    report["skills"] = int(df.skill.nunique())
    report["questions"] = int(df.question_id.nunique())
    report["positive_rate"] = round(float(df["correct"].mean()), 4)
    report["rows_per_week"] = {int(k): int(v) for k, v in df.groupby("week").size().items()}
    failed = [k for k, ok in report["checks"].items() if not ok]
    report["passed"] = not failed
    if failed:
        raise ValueError(f"data validation failed: {failed}")
    return report


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop nulls / duplicates, enforce dtypes, sort by time."""
    df = df.dropna(subset=REQUIRED).copy()
    df = df[df["correct"].isin([0, 1])]
    df = df.drop_duplicates(subset=["user_id", "question_id", "ts"])
    df["correct"] = df["correct"].astype(int)
    df["week"] = df["week"].astype(int)
    df = df.sort_values(["ts", "order_id"], kind="mergesort").reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(PROCESSED_DIR / "attempts.parquet"))
    args = ap.parse_args()
    t0 = time.time()
    p = load_params()
    if p["data"]["source"] == "assistments":
        n_weeks = p["batches"]["train_weeks"] + p["batches"]["n_live_batches"]
        df = load_assistments(p["data"]["assistments_path"], n_weeks)
    else:
        df = pd.read_parquet(RAW_DIR / "synthetic_attempts.parquet")
    df = clean(df)
    report = validate(df)
    report["seconds"] = round(time.time() - t0, 3)
    report["throughput_rows_per_s"] = round(len(df) / max(report["seconds"], 1e-6))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "data_quality.json").write_text(json.dumps(report, indent=2))
    log.info("ingested %d rows -> %s | passed=%s", len(df), args.out, report["passed"])


if __name__ == "__main__":
    main()
