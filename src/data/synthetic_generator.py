"""Synthetic Sanrathi-style quiz-attempt generator (with controllable drift).

Each student has an ability, a per-topic affinity and learns a little with every
attempt (so history genuinely predicts the next answer).  In the *drift week* two
things change, mimicking real EdTech incidents:

* new syllabus topics appear (students have no history on them), and
* every question becomes harder (``drift_difficulty_shift`` logits).

Output columns: user_id, skill, question_id, correct, ts, week, order_id
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import RAW_DIR, load_params, setup_logging

log = setup_logging("generator")

BASE_SKILLS = ["Fractions", "Decimals", "Percentages", "Ratios", "Algebra Basics",
               "Linear Equations", "Geometry Area", "Angles", "Probability",
               "Statistics", "Exponents", "Graphs", "Number Theory", "Mensuration"]
NEW_SKILLS = ["Quadratics", "Trigonometry", "Matrices", "Logarithms"]
WEEK_SECONDS = 7 * 24 * 3600
T0 = 1_700_000_000  # fixed epoch start => deterministic data


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_students=600, n_skills=12, questions_per_skill=25, n_weeks=7,
             attempts_per_week=12, drift_week=7, drift_new_skills=3,
             drift_new_skill_share=0.5, drift_difficulty_shift=1.0, seed=42) -> pd.DataFrame:
    """Create the attempts table.  Pure function => unit-testable and reproducible."""
    rng = np.random.default_rng(seed)
    skills = BASE_SKILLS[:n_skills]
    new_skills = NEW_SKILLS[:drift_new_skills] if drift_week else []
    all_skills = skills + new_skills

    # question bank: fixed difficulty per question
    q_diff = {}
    for s in all_skills:
        for q in range(questions_per_skill):
            q_diff[(s, q)] = rng.normal(0.0, 1.0)

    ability = rng.normal(0.0, 0.9, n_students)
    affinity = rng.normal(0.0, 0.5, (n_students, len(all_skills)))
    skill_pref = rng.dirichlet(np.ones(len(skills)) * 1.5, n_students)
    counts = np.zeros((n_students, len(all_skills)), dtype=int)

    rows = []
    for week in range(1, n_weeks + 1):
        is_drift = bool(drift_week) and week == drift_week
        for u in range(n_students):
            # students are not equally active: some skip weeks
            n_att = rng.poisson(attempts_per_week)
            if n_att == 0:
                continue
            times = np.sort(rng.uniform(0, WEEK_SECONDS, n_att)) + T0 + (week - 1) * WEEK_SECONDS
            for t in times:
                if is_drift and new_skills and rng.random() < drift_new_skill_share:
                    si = len(skills) + rng.integers(len(new_skills))
                else:
                    si = rng.choice(len(skills), p=skill_pref[u])
                s = all_skills[si]
                q = int(rng.integers(questions_per_skill))
                diff = q_diff[(s, q)] + (drift_difficulty_shift if is_drift else 0.0)
                learned = 0.06 * np.log1p(counts[u, si]) * 3
                logit = ability[u] + affinity[u, si] + learned - diff
                correct = int(rng.random() < _sigmoid(logit))
                counts[u, si] += 1
                rows.append((f"S{u:04d}", s, f"{s[:3].upper()}_{q:02d}", correct, float(t), week))
    df = pd.DataFrame(rows, columns=["user_id", "skill", "question_id", "correct", "ts", "week"])
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
    df["order_id"] = np.arange(len(df))
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(RAW_DIR / "synthetic_attempts.parquet"))
    args = ap.parse_args()
    p = load_params()
    g = {k: v for k, v in p["generator"].items()}
    df = generate(seed=p["data"]["seed"], **g)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    log.info("generated %d attempts, %d students, %d skills -> %s",
             len(df), df.user_id.nunique(), df.skill.nunique(), args.out)


if __name__ == "__main__":
    main()
