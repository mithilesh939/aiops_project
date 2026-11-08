"""Model registry helpers: register the best run, promote with a gate, read the production model.

Alias convention (Lecture 3 / MLflow Exercise 4): the alias ``production`` always points to
the version served by the API.  ``candidate`` marks the newest unpromoted version.
"""
from __future__ import annotations

import argparse
import time

import mlflow
from mlflow import MlflowClient

from src.common.config import load_params, settings, setup_logging

log = setup_logging("register")


def client() -> MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow_uri)
    return MlflowClient()


def production_version(name: str | None = None):
    """Return the ModelVersion currently aliased ``production`` (or None)."""
    name = name or settings.model_name
    try:
        return client().get_model_version_by_alias(name, settings.model_alias)
    except Exception:  # noqa: BLE001  (no such alias / model yet)
        return None


def register_run(run_id: str, name: str | None = None, tags: dict | None = None):
    """Register ``runs:/<run_id>/model`` as a new version of the registered model."""
    name = name or settings.model_name
    mv = mlflow.register_model(f"runs:/{run_id}/model", name)
    c = client()
    for k, v in (tags or {}).items():
        c.set_model_version_tag(name, mv.version, k, str(v))
    return mv


def promote(version: str, name: str | None = None, reason: str = "") -> None:
    """Point the production alias to ``version`` and record who/why as tags."""
    name = name or settings.model_name
    c = client()
    c.set_registered_model_alias(name, settings.model_alias, version)
    c.set_model_version_tag(name, version, "promoted_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    c.set_model_version_tag(name, version, "promotion_reason", reason)
    log.info("alias '%s' -> version %s (%s)", settings.model_alias, version, reason)


def register_best(min_gain_over_baseline: float = 0.02) -> str:
    """Pick the best run of the initial experiment, register it and promote it to production.

    Fails if the best model does not clearly beat the baseline (project acceptance criterion).
    """
    p = load_params()
    c = client()
    exp = c.get_experiment_by_name(p["registry"]["experiment"])
    runs = c.search_runs([exp.experiment_id], "tags.stage = 'initial'",
                         order_by=["attributes.start_time DESC"], max_results=50)
    # keep only the latest run per model kind
    latest = {}
    for r in runs:
        latest.setdefault(r.data.tags["model_kind"], r)
    base_auc = latest["baseline"].data.metrics["valid_auc"]
    best = max((r for k, r in latest.items() if k != "baseline"),
               key=lambda r: r.data.metrics["valid_auc"])
    best_auc = best.data.metrics["valid_auc"]
    if best_auc < base_auc + min_gain_over_baseline:
        raise RuntimeError(f"best AUC {best_auc:.3f} does not beat baseline {base_auc:.3f}")
    mv = register_run(best.info.run_id, tags={
        "valid_auc": round(best_auc, 4), "model_kind": best.data.tags["model_kind"],
        "git_commit": best.data.tags.get("git_commit", "")})
    promote(mv.version, reason=f"initial best model, AUC {best_auc:.4f} vs baseline {base_auc:.4f}")
    return mv.version


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-gain", type=float, default=0.02)
    args = ap.parse_args()
    v = register_best(args.min_gain)
    log.info("registered and promoted version %s", v)


if __name__ == "__main__":
    main()
