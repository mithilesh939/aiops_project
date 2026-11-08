"""Central configuration: project paths, params.yaml loader and environment settings.

Every stage (DVC, training, API, monitor) reads its numbers from ``params.yaml`` so
that a single file documents (and versions) all tunable behaviour.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("QUIZPULSE_HOME", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BATCH_DIR = DATA_DIR / "batches"
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", ROOT / "reports"))

FEATURES = [
    "overall_acc",
    "skill_acc",
    "skill_attempts",
    "recent_acc5",
    "hours_since_last",
    "question_difficulty",
]
TARGET = "correct"


@lru_cache(maxsize=1)
def load_params(path: str | None = None) -> dict[str, Any]:
    """Load params.yaml (cached)."""
    p = Path(path) if path else ROOT / "params.yaml"
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def setup_logging(name: str = "quizpulse") -> logging.Logger:
    """Configure one consistent log format for every component."""
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


class Settings:
    """Runtime settings taken from environment variables (12-factor style)."""

    @property
    def database_url(self) -> str:
        return os.environ.get("DATABASE_URL", f"sqlite:///{ROOT / 'quizpulse.db'}")

    @property
    def mlflow_uri(self) -> str:
        return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")

    @property
    def model_name(self) -> str:
        return os.environ.get("MODEL_NAME", load_params()["registry"]["model_name"])

    @property
    def model_alias(self) -> str:
        return os.environ.get("MODEL_ALIAS", "production")

    @property
    def monitor_url(self) -> str:
        return os.environ.get("MONITOR_URL", "http://localhost:8100")


settings = Settings()
