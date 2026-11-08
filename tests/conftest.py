"""Shared fixtures: temp SQLite DB, tiny synthetic dataset and a fake model (no MLflow needed)."""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_tmp = tempfile.mkdtemp(prefix="qp_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_tmp}/mlflow.db"
os.environ["QP_LOAD_MODEL"] = "0"
os.environ["REPORTS_DIR"] = _tmp

from src.common import db  # noqa: E402
from src.data.synthetic_generator import generate  # noqa: E402


class FakeModel:
    """Deterministic stand-in for the registry model: p grows with skill accuracy."""

    def predict(self, frame):
        p = 0.2 + 0.6 * frame["skill_acc"].to_numpy(dtype=float) * 0.5 + 0.3 * frame["question_difficulty"].to_numpy(dtype=float)
        return np.clip(p, 0.01, 0.99)


@pytest.fixture(scope="session")
def small_df():
    return generate(n_students=40, n_skills=5, questions_per_skill=8, n_weeks=4,
                    attempts_per_week=8, drift_week=0, seed=1)


@pytest.fixture(scope="session")
def seeded_db(small_df):
    """Insert the first 3 weeks as history into the (temp) attempts table."""
    db.reset_engine()
    eng = db.get_engine()
    hist = small_df[small_df.week <= 3][["user_id", "skill", "question_id", "correct", "ts"]].copy()
    hist["batch"] = 0
    hist.to_sql("attempts", eng, if_exists="append", index=False)
    return eng


@pytest.fixture()
def client(seeded_db):
    from fastapi.testclient import TestClient

    from src.serving import mastery
    from src.serving.app import app
    from src.serving.model_loader import holder
    mastery._catalog_cache.update(t=0.0, df=None)
    mastery._clock_cache.update(t=0.0, last=None)
    holder.set_model(FakeModel(), "1")
    with TestClient(app) as c:
        yield c
