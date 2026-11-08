"""Simulate live traffic: replay a batch through the API (/predict then /outcomes).

Examples
--------
python scripts/replay_batches.py --batch 1                 # one normal "week"
python scripts/replay_batches.py --batch 1 2 3             # several weeks
python scripts/replay_batches.py --batch 4                 # the DRIFTED week
python scripts/replay_batches.py --batch 1 --student-fraction 0.4   # 40% of the students, faster
python scripts/replay_batches.py --batch 1 --limit 500 --rate 50   # first 500 rows at 50 req/s
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import BATCH_DIR, setup_logging  # noqa: E402

log = setup_logging("replay")


def replay(batch: int, api: str, limit: int | None, rate: float | None, session: requests.Session,
           student_fraction: float = 1.0) -> dict:
    df = pd.read_parquet(BATCH_DIR / f"batch_{batch}.parquet").sort_values(["ts", "order_id"])
    if student_fraction < 1.0:      # keep ALL rows of a fixed random subset of students (realistic sequences)
        users = df["user_id"].drop_duplicates().sample(frac=student_fraction, random_state=7)
        df = df[df["user_id"].isin(set(users))]
    if limit:
        df = df.head(limit)
    t0, ok, fail = time.time(), 0, 0
    for i, r in enumerate(df.itertuples(), start=1):
        try:
            p = session.post(f"{api}/predict", timeout=30, json={
                "student_id": r.user_id, "question_id": r.question_id, "skill": r.skill,
                "timestamp": float(r.ts), "batch": batch})
            p.raise_for_status()
            o = session.post(f"{api}/outcomes", timeout=30, json={
                "prediction_id": p.json()["prediction_id"], "correct": int(r.correct)})
            o.raise_for_status()
            ok += 1
        except requests.RequestException as exc:
            fail += 1
            if fail <= 3:
                log.warning("request failed: %s", exc)
        if rate:
            time.sleep(1.0 / rate)
        if i % 1000 == 0:
            log.info("batch %d: %d/%d rows (%.0f rows/s)", batch, i, len(df), i / (time.time() - t0))
    secs = time.time() - t0
    log.info("batch %d done: ok=%d failed=%d in %.1fs (%.0f rows/s)", batch, ok, fail, secs, ok / max(secs, 1e-9))
    return {"batch": batch, "ok": ok, "failed": fail, "seconds": round(secs, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, nargs="+", required=True)
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--limit", type=int, default=None, help="max rows per batch")
    ap.add_argument("--student-fraction", type=float, default=1.0,
                    help="replay only this fraction of students (all their rows) - faster demos")
    ap.add_argument("--rate", type=float, default=None, help="max rows per second (default: as fast as possible)")
    args = ap.parse_args()
    s = requests.Session()
    for b in args.batch:
        if not (BATCH_DIR / f"batch_{b}.parquet").exists():
            sys.exit(f"batch_{b}.parquet not found - run the data pipeline first")
        replay(b, args.api, args.limit, args.rate, s, args.student_fraction)


if __name__ == "__main__":
    main()
