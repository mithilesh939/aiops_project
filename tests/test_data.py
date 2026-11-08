"""Data generation, validation and time-based batching."""
import numpy as np
import pandas as pd
import pytest

from src.data.ingest import clean, validate
from src.data.make_batches import assign_batches
from src.data.synthetic_generator import generate


def test_generator_is_deterministic():
    kw = dict(n_students=20, n_skills=4, questions_per_skill=5, n_weeks=3, attempts_per_week=5, drift_week=0)
    pd.testing.assert_frame_equal(generate(seed=5, **kw), generate(seed=5, **kw))


def test_drift_week_introduces_new_skills_and_harder_questions():
    kw = dict(n_students=150, n_skills=5, questions_per_skill=8, n_weeks=4, attempts_per_week=10, seed=2)
    d = generate(drift_week=4, drift_new_skills=2, **kw)
    base = set(d[d.week < 4].skill)
    assert len(set(d[d.week == 4].skill) - base) == 2
    no_shift = generate(drift_week=0, **kw)
    assert d[d.week == 4].correct.mean() < no_shift[no_shift.week == 4].correct.mean()


def test_validate_passes_on_good_data(small_df):
    rep = validate(clean(small_df.assign(order_id=range(len(small_df)))))
    assert rep["passed"] and rep["students"] == 40


def test_validate_rejects_non_binary_target():
    bad = pd.DataFrame({"user_id": ["a"] * 1500, "skill": "s", "question_id": "q", "correct": 2,
                        "ts": np.arange(1500.0), "week": 1, "order_id": np.arange(1500)})
    with pytest.raises(ValueError):
        validate(bad)


def test_batches_are_time_ordered_and_disjoint(small_df):
    b = assign_batches(small_df, train_weeks=2, n_live=2)
    assert set(b.batch) == {0, 1, 2}
    assert b[b.batch == 0].ts.max() < b[b.batch == 1].ts.min() < b[b.batch == 2].ts.min()
    assert len(b) == len(small_df)
