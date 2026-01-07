"""MLflow helpers"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

try:
    import mlflow
except ModuleNotFoundError:
    mlflow = None


def is_mlflow_available() -> bool:
    return mlflow is not None


def setup_mlflow(tracking_uri: str = "file:mlruns", experiment: str = "pm-risk") -> None:
    if mlflow is None:
        return

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)


def log_params(params: dict) -> None:
    if mlflow is None:
        return

    mlflow.log_params(params)


def log_metrics(metrics: dict, step: int | None = None) -> None:
    if mlflow is None:
        return

    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str | Path, artifact_path: str | None = None) -> None:
    if mlflow is None:
        return

    mlflow.log_artifact(str(path), artifact_path=artifact_path)


@contextmanager
def start_run(run_name: str | None = None, tags: dict | None = None):
    if mlflow is None:
        yield
        return

    with mlflow.start_run(run_name=run_name, tags=tags):
        yield
