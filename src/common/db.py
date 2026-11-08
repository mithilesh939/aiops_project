"""Database layer (SQLAlchemy).  Postgres in Docker, SQLite for tests/local runs.

Tables
------
attempts      every answered question (history used for online features)
predictions   every /predict call with the feature values that were used
outcomes      delayed labels: what the student actually answered
pipeline_runs audit trail of ingest / train / drift / retrain / promotion events
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Column, Float, Index, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.common.config import settings

Base = declarative_base()


class Attempt(Base):
    __tablename__ = "attempts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(32), index=True, nullable=False)
    skill = Column(String(64), nullable=False)
    question_id = Column(String(64), nullable=False)
    correct = Column(Integer, nullable=False)
    ts = Column(Float, nullable=False)          # unix seconds of the attempt
    batch = Column(Integer, default=0)
    __table_args__ = (Index("ix_attempt_q_ts", "question_id", "ts"),
                      Index("ix_attempt_u_ts", "user_id", "ts"))


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(32), index=True)
    skill = Column(String(64))
    question_id = Column(String(64))
    p_correct = Column(Float)
    model_version = Column(String(16))
    ts = Column(Float)
    batch = Column(Integer, default=-1)
    overall_acc = Column(Float)
    skill_acc = Column(Float)
    skill_attempts = Column(Float)
    recent_acc5 = Column(Float)
    hours_since_last = Column(Float)
    question_difficulty = Column(Float)


class Outcome(Base):
    __tablename__ = "outcomes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, index=True)
    user_id = Column(String(32))
    correct = Column(Integer)
    ts = Column(Float)


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    kind = Column(String(32), index=True)       # ingest|train|drift|retrain|promotion|replay
    status = Column(String(16))                 # running|success|failed|skipped
    started = Column(Float)
    finished = Column(Float)
    duration_s = Column(Float)
    rows = Column(BigInteger)
    details = Column(Text)                      # JSON string
    mlflow_run_id = Column(String(64))


_engine = None
_Session = None


def get_engine():
    """Create (once) and return the engine, creating the tables if needed."""
    global _engine, _Session
    if _engine is None:
        url = settings.database_url
        kwargs = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        _engine = create_engine(url, **kwargs)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False)
        Base.metadata.create_all(_engine)
    return _engine


def reset_engine() -> None:
    """Forget the cached engine (used by tests that switch DATABASE_URL)."""
    global _engine, _Session
    _engine, _Session = None, None


@contextmanager
def session_scope():
    """Transactional session context manager."""
    get_engine()
    session = _Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def wait_for_db(retries: int = 40, delay: float = 1.5) -> None:
    """Block until the database accepts connections (Docker start-up ordering)."""
    last = None
    for _ in range(retries):
        try:
            get_engine().connect().close()
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            reset_engine()
            time.sleep(delay)
    raise RuntimeError(f"database not reachable: {last}")


def log_run(kind: str, status: str, started: float, rows: int = 0,
            details: dict | None = None, mlflow_run_id: str | None = None) -> int:
    """Insert one row in pipeline_runs and return its id."""
    now = datetime.now(timezone.utc).timestamp()
    with session_scope() as s:
        row = PipelineRun(kind=kind, status=status, started=started, finished=now,
                          duration_s=round(now - started, 3), rows=rows,
                          details=json.dumps(details or {}, default=str),
                          mlflow_run_id=mlflow_run_id)
        s.add(row)
        s.flush()
        return row.id
