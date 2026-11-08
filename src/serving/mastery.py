"""Topic mastery, weak-topic detection and adaptive question selection.

Mastery of a topic = average model-predicted P(correct) of the student on a sample of
that topic's questions *right now*.  A topic is weak when mastery < ``WEAK_THRESHOLD``.
"""
from __future__ import annotations

import time

import pandas as pd
from sqlalchemy import text

from src.common.config import FEATURES, load_params
from src.common.db import get_engine
from src.features.build_features import online_features
from src.serving.model_loader import holder

WEAK_THRESHOLD = 0.5
QUESTIONS_PER_SKILL = 6
_catalog_cache: dict = {"t": 0.0, "df": None}
_clock_cache: dict = {"t": 0.0, "last": None}


def question_catalog(ttl: float = 60.0) -> pd.DataFrame:
    """All known (skill, question_id) pairs with their attempt statistics (cached)."""
    if _catalog_cache["df"] is None or time.time() - _catalog_cache["t"] > ttl:
        q = ("SELECT skill, question_id, COUNT(*) AS n, SUM(correct) AS s "
             "FROM attempts GROUP BY skill, question_id")
        _catalog_cache["df"] = pd.read_sql(text(q), get_engine())
        _catalog_cache["t"] = time.time()
    return _catalog_cache["df"]


def platform_now(ttl: float = 15.0) -> float:
    """The platform's logical 'now': wall clock, but never more than 1h after the last recorded
    attempt.  (With replayed/synthetic data from the past this keeps ``hours_since_last`` realistic.)"""
    if time.time() - _clock_cache["t"] > ttl:
        with get_engine().connect() as c:
            last = c.execute(text("SELECT MAX(ts) FROM attempts")).scalar()
        _clock_cache.update(t=time.time(), last=float(last) if last else None)
    last = _clock_cache["last"]
    return min(time.time(), last + 3600) if last else time.time()


def student_history(student_id: str, before: float | None = None) -> pd.DataFrame:
    q = "SELECT skill, question_id, correct, ts FROM attempts WHERE user_id = :u"
    params = {"u": student_id}
    if before is not None:
        q += " AND ts < :t"
        params["t"] = before
    q += " ORDER BY ts"
    return pd.read_sql(text(q), get_engine(), params=params)


def question_stats(question_id: str, before: float) -> tuple[float, float]:
    with get_engine().connect() as c:
        row = c.execute(text("SELECT COALESCE(SUM(correct),0), COUNT(*) FROM attempts "
                             "WHERE question_id = :q AND ts < :t"),
                        {"q": question_id, "t": before}).fetchone()
    return float(row[0]), float(row[1])


def _score_candidates(student_id: str, now: float, cands: pd.DataFrame, hist: pd.DataFrame):
    """Predict P(correct) for each candidate question (one batched model call)."""
    params = load_params()
    rows = []
    for r in cands.itertuples():
        f = online_features(hist, student_id, r.skill, r.question_id, now, float(r.s), float(r.n), params)
        rows.append({**f, "skill": r.skill, "question_id": r.question_id})
    frame = pd.DataFrame(rows)
    frame["p"] = holder.predict(frame)
    return frame


def student_mastery(student_id: str, now: float | None = None) -> list[dict]:
    """Per-topic mastery for one student, weakest first."""
    now = now or platform_now()
    hist = student_history(student_id)
    cat = question_catalog()
    if cat.empty:
        return []
    cands = (cat.sort_values(["skill", "n"], ascending=[True, False])
                .groupby("skill").head(QUESTIONS_PER_SKILL))
    scored = _score_candidates(student_id, now, cands, hist)
    out = []
    for skill, g in scored.groupby("skill"):
        h = hist[hist["skill"] == skill]
        out.append({
            "skill": skill, "mastery": round(float(g["p"].mean()), 4), "attempts": int(len(h)),
            "observed_accuracy": round(float(h["correct"].mean()), 4) if len(h) else None,
            "weak": bool(g["p"].mean() < WEAK_THRESHOLD),
        })
    return sorted(out, key=lambda d: d["mastery"])


def next_questions(student_id: str, n: int, now: float | None = None) -> list[dict]:
    """Pick unseen questions from the weakest topics, aiming at p~0.6 (productive difficulty)."""
    now = now or platform_now()
    hist = student_history(student_id)
    mastery = student_mastery(student_id, now)
    if not mastery:
        return []
    weak = [m["skill"] for m in mastery[:3]]
    cat = question_catalog()
    seen = set(hist["question_id"])
    cands = cat[cat["skill"].isin(weak) & ~cat["question_id"].isin(seen)]
    if cands.empty:
        cands = cat[cat["skill"].isin(weak)]
    scored = _score_candidates(student_id, now, cands, hist)
    scored["gap"] = (scored["p"] - 0.6).abs()
    scored = scored.sort_values("gap")
    picked, per_skill = [], {}
    for r in scored.itertuples():           # spread over the weak topics
        if per_skill.get(r.skill, 0) >= max(1, -(-n // len(weak))):
            continue
        per_skill[r.skill] = per_skill.get(r.skill, 0) + 1
        picked.append({"question_id": r.question_id, "skill": r.skill, "p_correct": round(float(r.p), 4),
                       "reason": f"weak topic ({r.skill}); expected success {r.p:.0%} - challenging but doable"})
        if len(picked) == n:
            break
    return picked


assert FEATURES  # imported for type parity with the model contract
