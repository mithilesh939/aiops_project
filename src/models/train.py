"""Train and compare baseline / logistic regression / LightGBM, tracked in MLflow.

Run via DVC (``dvc repro``), via ``mlflow run .`` (MLproject) or directly:
    python -m src.models.train

Every run logs: params, validation + test metrics, the git commit, the data version
(hash of the feature file), params.yaml, a feature-importance plot, a ROC curve, a
reference sample of training features (used later for drift detection) and the
model itself as an MLflow pyfunc with a signature and input example.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.common.config import FEATURES, PROCESSED_DIR, REPORTS_DIR, ROOT, TARGET, load_params, settings, setup_logging
from src.models.evaluate import evaluate, time_split
from src.models.kt_wrapper import KTModel, SkillMeanBaseline

log = setup_logging("train")
INPUT_COLS = FEATURES + ["skill"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "no-git"


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def build_estimator(kind: str, p: dict):
    if kind == "logreg":
        lr = p["train"]["logreg"]
        return make_pipeline(StandardScaler(), LogisticRegression(C=lr["C"], max_iter=lr["max_iter"]))
    if kind == "lightgbm":
        return LGBMClassifier(**p["train"]["lightgbm"], random_state=p["train"]["seed"], verbose=-1)
    return SkillMeanBaseline()


def fit_estimator(kind: str, est, train: pd.DataFrame):
    if kind == "baseline":
        return est.fit(train["skill"], train[TARGET])
    est.fit(train[FEATURES].astype(float), train[TARGET])
    return est


def predict(kind: str, est, df: pd.DataFrame) -> np.ndarray:
    if kind == "baseline":
        return est.predict_proba(df)[:, 1]
    return est.predict_proba(df[FEATURES].astype(float))[:, 1]


def log_plots(kind, est, valid, p_valid):
    """ROC curve + feature importance as run artifacts."""
    fpr, tpr, _ = roc_curve(valid[TARGET], p_valid)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=kind)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (validation)")
    ax.legend()
    mlflow.log_figure(fig, "plots/roc_curve.png")
    plt.close(fig)
    imp = None
    if kind == "lightgbm":
        imp = est.feature_importances_
    elif kind == "logreg":
        imp = np.abs(est[-1].coef_[0])
    if imp is not None:
        s = pd.Series(imp, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        s.plot.barh(ax=ax)
        ax.set_title("Feature importance")
        fig.tight_layout()
        mlflow.log_figure(fig, "plots/feature_importance.png")
        plt.close(fig)
        mlflow.log_dict(s.sort_values(ascending=False).round(4).to_dict(), "feature_importance.json")


def train_one(kind: str, train, valid, test, p: dict, data_version: str, experiment: str,
              stage: str = "initial", extra_tags: dict | None = None,
              reference: pd.DataFrame | None = None) -> dict:
    """Train one model inside one MLflow run; returns metrics + run id."""
    with mlflow.start_run(run_name=f"{kind}-{stage}") as run:
        mlflow.set_tags({"model_kind": kind, "stage": stage, "git_commit": git_commit(),
                         "data_version": data_version, **(extra_tags or {})})
        mlflow.log_params({"model_kind": kind, "n_train": len(train), "n_valid": len(valid),
                           "n_test": len(test), "features": ",".join(FEATURES),
                           "valid_fraction": p["train"]["valid_fraction"]})
        if kind in p["train"]:
            mlflow.log_params({f"{kind}.{k}": v for k, v in p["train"][kind].items()})
        t0 = time.time()
        est = fit_estimator(kind, build_estimator(kind, p), train)
        mlflow.log_metric("train_seconds", time.time() - t0)

        p_valid, p_test = predict(kind, est, valid), predict(kind, est, test)
        m_valid, m_test = evaluate(valid[TARGET], p_valid), evaluate(test[TARGET], p_test)
        for k, v in m_valid.items():
            mlflow.log_metric(f"valid_{k}", v)
        for k, v in m_test.items():
            mlflow.log_metric(f"test_{k}", v)
        log_plots(kind, est, valid, p_valid)

        # drift reference + params + model
        # reference = the most recent 40% of the training period (histories are "matured", so
        # cumulative features are comparable with what the API sees later)
        # (a retrain passes its own reference = the regime that triggered it)
        if reference is None:
            reference = train.sort_values(["ts", "order_id"]).iloc[int(len(train) * 0.6):]
        ref = reference[INPUT_COLS].sample(min(5000, len(reference)), random_state=0)
        ref_path = PROCESSED_DIR / "reference_features.parquet"
        ref.to_parquet(ref_path, index=False)
        mlflow.log_artifact(str(ref_path), "reference")
        mlflow.log_artifact(str(ROOT / "params.yaml"), "config")

        sample = valid[INPUT_COLS].head(5).astype({f: float for f in FEATURES})
        signature = infer_signature(sample, predict(kind, est, valid.head(5)))
        mlflow.pyfunc.log_model(
            name="model", python_model=KTModel(est, kind), signature=signature,
            input_example=sample, code_paths=[str(ROOT / "src")],
            pip_requirements=["mlflow", "scikit-learn", "lightgbm", "pandas", "numpy"])
        log.info("%-9s valid AUC=%.4f logloss=%.4f | test AUC=%.4f | run %s", kind,
                 m_valid["auc"], m_valid["logloss"], m_test["auc"], run.info.run_id[:8])
        return {"kind": kind, "run_id": run.info.run_id, "valid": m_valid, "test": m_test}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=str(PROCESSED_DIR / "features.parquet"))
    ap.add_argument("--models", default="baseline,logreg,lightgbm")
    args = ap.parse_args()
    p = load_params()
    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(p["registry"]["experiment"])

    df = pd.read_parquet(args.features)
    batch0 = df[df["batch"] == 0]
    train, valid = time_split(batch0, p["train"]["valid_fraction"])
    test = df[df["batch"] == 1]                      # next "week", no drift => honest hold-out
    data_version = file_hash(Path(args.features))
    log.info("train=%d valid=%d test=%d data_version=%s", len(train), len(valid), len(test), data_version)

    results = []
    for kind in args.models.split(","):
        results.append(train_one(kind.strip(), train, valid, test, p, data_version,
                                 p["registry"]["experiment"]))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    # flat file for `dvc metrics show` / `dvc metrics diff`
    flat = {r["kind"]: {"valid_auc": round(r["valid"]["auc"], 4), "valid_logloss": round(r["valid"]["logloss"], 4),
                        "valid_accuracy": round(r["valid"]["accuracy"], 4), "test_auc": round(r["test"]["auc"], 4)}
            for r in results}
    (REPORTS_DIR / "train_metrics.json").write_text(json.dumps(flat, indent=2))
    best = max(results, key=lambda r: r["valid"]["auc"])
    log.info("best model: %s (valid AUC %.4f)", best["kind"], best["valid"]["auc"])


if __name__ == "__main__":
    main()
