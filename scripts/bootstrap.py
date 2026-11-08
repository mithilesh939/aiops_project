"""One-shot start-up job (Docker Compose service ``bootstrap``).

1. wait for Postgres and the MLflow server
2. build the data/feature pipeline if the files are missing (normally `dvc repro` made them)
3. train + register the first production model if the registry has none
4. seed the students' history into the attempts table
Idempotent: safe to run on every `docker compose up`.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlflow  # noqa: E402

from src.common.config import BATCH_DIR, PROCESSED_DIR, settings, setup_logging  # noqa: E402
from src.common.db import log_run, wait_for_db  # noqa: E402
from src.models.register import production_version  # noqa: E402

log = setup_logging("bootstrap")


def sh(*cmd: str) -> None:
    log.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def wait_for_mlflow(retries: int = 60) -> None:
    mlflow.set_tracking_uri(settings.mlflow_uri)
    for _ in range(retries):
        try:
            mlflow.search_experiments()
            return
        except Exception:  # noqa: BLE001
            time.sleep(2)
    raise RuntimeError("MLflow server not reachable")


def main() -> None:
    wait_for_db()
    wait_for_mlflow()
    if not (BATCH_DIR / "batch_0.parquet").exists() or not (PROCESSED_DIR / "features.parquet").exists():
        t0 = time.time()
        log.info("data files missing -> running the data pipeline inside the container")
        sh("python", "-m", "src.data.synthetic_generator")
        sh("python", "-m", "src.data.ingest")
        sh("python", "-m", "src.data.make_batches")
        sh("python", "-m", "src.features.build_features", "--engine", "pandas")
        log_run("ingest", "success", t0, details={"note": "built by bootstrap"})
    if production_version() is None:
        t0 = time.time()
        sh("python", "-m", "src.models.train")
        log_run("train", "success", t0, details={"models": "baseline,logreg,lightgbm"})
        t1 = time.time()
        sh("python", "-m", "src.models.register")
        log_run("register", "success", t1, details={"alias": "production"})
    else:
        log.info("production model already registered - skipping training")
    sh("python", "scripts/seed_history.py")
    log.info("bootstrap complete")


if __name__ == "__main__":
    main()
