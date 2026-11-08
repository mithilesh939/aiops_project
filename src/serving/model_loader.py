"""Loads the ``production`` model from the MLflow registry and hot-reloads on promotion."""
from __future__ import annotations

import threading
import time

import mlflow
import mlflow.pyfunc
import pandas as pd

from src.common.config import FEATURES, settings, setup_logging
from src.serving import metrics

log = setup_logging("model_loader")


class ModelHolder:
    """Thread-safe holder of the currently served model."""

    def __init__(self):
        self.model = None
        self.version: str | None = None
        self.loaded_at: float | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def set_model(self, model, version: str) -> None:
        """Install a model (also used by tests to inject a fake)."""
        with self._lock:
            self.model, self.version, self.loaded_at = model, str(version), time.time()
        metrics.MODEL_LOADED.set(1)
        metrics.MODEL_VERSION.set(float(version) if str(version).isdigit() else 0)
        metrics.MODEL_RELOADS.inc()

    def load_from_registry(self) -> bool:
        """Load models:/<name>@production.  Returns True if a (new) model was loaded."""
        mlflow.set_tracking_uri(settings.mlflow_uri)
        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias(settings.model_name, settings.model_alias)
        if self.version == str(mv.version):
            return False
        uri = f"models:/{settings.model_name}@{settings.model_alias}"
        log.info("loading %s (version %s)", uri, mv.version)
        model = mlflow.pyfunc.load_model(uri)
        self.set_model(model, mv.version)
        log.info("now serving version %s", mv.version)
        return True

    def predict(self, frame: pd.DataFrame):
        if not self.loaded:
            raise RuntimeError("no model loaded")
        frame = frame.copy()
        frame[FEATURES] = frame[FEATURES].astype(float)
        with self._lock:
            model = self.model
        return model.predict(frame[FEATURES + ["skill"]])

    def watch(self, interval: float = 10.0) -> threading.Thread:
        """Background thread: poll the registry alias and hot-swap the model on change."""
        def loop():
            while True:
                try:
                    self.load_from_registry()
                except Exception as exc:  # noqa: BLE001
                    log.warning("registry poll failed: %s", exc)
                time.sleep(interval)
        t = threading.Thread(target=loop, daemon=True, name="model-watcher")
        t.start()
        return t


holder = ModelHolder()
