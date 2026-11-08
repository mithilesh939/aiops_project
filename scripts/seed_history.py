"""Load batch_0 (the training weeks) into the ``attempts`` table = the students' history.

The API builds online features from this table, so it must contain everything the model
was trained on.  Idempotent: does nothing if the table is already filled.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import BATCH_DIR, setup_logging  # noqa: E402
from src.common.db import get_engine, log_run, wait_for_db  # noqa: E402

log = setup_logging("seed")


def main() -> None:
    wait_for_db()
    t0 = time.time()
    eng = get_engine()
    with eng.connect() as c:
        n = c.exec_driver_sql("SELECT COUNT(*) FROM attempts").scalar()
    if n:
        log.info("attempts table already has %d rows - skipping seed", n)
        return
    df = pd.read_parquet(BATCH_DIR / "batch_0.parquet")
    out = df[["user_id", "skill", "question_id", "correct", "ts", "batch"]].copy()
    out["correct"] = out["correct"].astype(int)
    out.to_sql("attempts", eng, if_exists="append", index=False, chunksize=5000, method="multi")
    log_run("ingest", "success", t0, rows=len(out), details={"source": "batch_0", "table": "attempts"})
    log.info("seeded %d historical attempts in %.1fs", len(out), time.time() - t0)


if __name__ == "__main__":
    main()
