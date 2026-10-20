"""Experiment tracking and model registry with MLflow (stages 5 and 8).

Every training run is logged: parameters, cross-validation scores for each
tuning candidate (as nested runs), validation and test metrics, figures and
the fitted model. The champion is registered as a new version of the
`fraudlens-champion` model and given the alias `champion`.

Tracking URI:
  * MLFLOW_TRACKING_URI or config platform.mlflow_tracking_uri if set
    (e.g. http://localhost:5000 for the Docker MLflow server);
  * otherwise a local SQLite store at mlruns/mlflow.db (view it with
    `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`).

If MLflow is not installed, every call is a no-op so the pipeline still runs.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

from .utils import get_logger

log = get_logger()


class Tracker:
    def __init__(self, cfg: dict, enabled: bool = True):
        self.cfg = cfg
        self.enabled = enabled
        self.mlflow = None
        self.run_id = None
        if not enabled:
            return
        try:
            import mlflow
        except ImportError:
            log.warning("MLflow not installed; experiment tracking disabled")
            return
        self.mlflow = mlflow
        uri = cfg["platform"].get("mlflow_tracking_uri")
        if not uri:
            store = cfg["root"] / "mlruns"
            store.mkdir(exist_ok=True)
            uri = f"sqlite:///{(store / 'mlflow.db').as_posix()}"
            self.artifact_location = (store / "artifacts").resolve().as_uri()
        else:
            self.artifact_location = None
        mlflow.set_tracking_uri(uri)
        name = cfg["platform"].get("mlflow_experiment", "fraudlens")
        try:
            if mlflow.get_experiment_by_name(name) is None:
                mlflow.create_experiment(name, artifact_location=self.artifact_location)
            mlflow.set_experiment(name)
        except Exception as exc:  # unreachable server etc.
            log.warning("MLflow unavailable at %s (%s); tracking disabled", uri, exc)
            self.mlflow = None
        self.uri = uri

    @property
    def active(self) -> bool:
        return self.mlflow is not None

    @contextlib.contextmanager
    def run(self, name: str, tags: dict | None = None):
        if not self.active:
            yield None
            return
        with self.mlflow.start_run(run_name=name, tags=tags or {}) as r:
            self.run_id = r.info.run_id
            yield r

    def log_params(self, params: dict, prefix: str = "") -> None:
        if self.active:
            flat = {f"{prefix}{k}": (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in params.items()}
            self.mlflow.log_params({k: str(v)[:500] for k, v in flat.items()})

    def log_metrics(self, metrics: dict, prefix: str = "") -> None:
        if self.active:
            clean = {f"{prefix}{k}": float(v) for k, v in metrics.items()
                     if isinstance(v, (int, float)) and v is not None and v == v}
            self.mlflow.log_metrics(clean)

    def log_artifact(self, path: Path, folder: str | None = None) -> None:
        if self.active and Path(path).exists():
            self.mlflow.log_artifact(str(path), artifact_path=folder)

    def log_candidate(self, model_name: str, idx: int, params: dict, res: dict) -> None:
        if not self.active:
            return
        with self.mlflow.start_run(run_name=f"tune-{model_name}-{idx}", nested=True):
            self.mlflow.set_tags({"stage": "tuning", "model": model_name})
            self.log_params(params)
            self.mlflow.log_metrics({"cv_pr_auc_mean": res["cv_pr_auc_mean"], "cv_pr_auc_std": res["cv_pr_auc_std"]})

    def log_model(self, model, name: str, library: str, register_as: str | None = None) -> str | None:
        """Log a fitted model with its native MLflow flavour; optionally register it."""
        if not self.active:
            return None
        flavour = {"xgboost": self.mlflow.xgboost, "lightgbm": self.mlflow.lightgbm}.get(library, self.mlflow.sklearn)
        kwargs = {"registered_model_name": register_as} if register_as else {}
        if flavour is self.mlflow.sklearn:
            # MLflow 3 defaults to skops for scikit-learn; cloudpickle needs no extra dependency
            kwargs["serialization_format"] = "cloudpickle"
        attempts = [dict(name=name, **kwargs), dict(artifact_path=name, **kwargs),   # MLflow 3 / MLflow 2
                    dict(artifact_path=name, **{k: v for k, v in kwargs.items() if k != "serialization_format"})]
        for kw in attempts:
            try:
                info = flavour.log_model(model, **kw)
                return getattr(info, "model_uri", None)
            except TypeError:
                continue
            except Exception as exc:  # tracking must never break training
                log.warning("Could not log the model to MLflow (%s: %s)", type(exc).__name__, exc)
                return None
        return None

    def set_champion_alias(self, registered_model: str) -> int | None:
        """Point the `champion` alias at the newest registered version."""
        if not self.active:
            return None
        from mlflow import MlflowClient
        client = MlflowClient()
        try:
            versions = client.search_model_versions(f"name='{registered_model}'")
        except Exception as exc:
            log.warning("Model registry unavailable (%s)", exc)
            return None
        if not versions:
            return None
        latest = max(versions, key=lambda v: int(v.version))
        client.set_registered_model_alias(registered_model, "champion", latest.version)
        return int(latest.version)
