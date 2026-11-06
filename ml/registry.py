"""ParkWatch — model registry (MLflow of the Azure ML workspace).

Every trained model is registered under MODEL_NAME; the champion is the version tagged
champion=true. The Azure ML registry does not support MLflow aliases (it returns 404,
checked on 2026-11-02), hence the tag. This is the only place that knows how the champion
is marked: promote.py and inference use it.

The registry always lives in Azure ML, even when the job tracks to the self-hosted MLflow
(ml/tracking.py): every function runs inside tracking.azureml(), which is a no-op when the
global MLflow already points to the workspace (inference, local runs).
"""
from __future__ import annotations

from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

import tracking
from azure_utils import retry

MODEL_NAME = "parknet"
CHAMPION_TAG = "champion"


def find_champion() -> ModelVersion | None:
    """Version tagged as champion; if a half-finished promotion left several, the newest one."""
    with tracking.azureml():
        client = MlflowClient()
        found = retry(
            lambda: client.search_model_versions(f"name='{MODEL_NAME}' and tag.{CHAMPION_TAG}='true'"),
            "find champion",
        )
    return max(found, key=lambda m: int(m.version)) if found else None


def register(model_dir: Path, tags: dict) -> ModelVersion:
    """Uploads the model folder as an artifact of the Azure ML job run and registers it as a new version.

    Outside a job (no Azure ML run) the current fluent run is used, as before.
    """
    run_id = tracking.AZUREML_RUN_ID or (mlflow.active_run() or mlflow.start_run()).info.run_id
    with tracking.azureml():
        client = MlflowClient()
        retry(lambda: client.log_artifacts(run_id, str(model_dir), "model"), "upload model")
        version = retry(lambda: mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME), "register model")
        for key, value in tags.items():
            retry(lambda: client.set_model_version_tag(MODEL_NAME, version.version, key, str(value)), f"tag {key}")
    return version


def download(version: ModelVersion, dst: Path) -> Path:
    """Local folder with parknet.pt + model_config.json of a registered version."""
    with tracking.azureml():
        path = retry(
            lambda: mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}/{version.version}", dst_path=str(dst)),
            f"download {MODEL_NAME}:{version.version}",
        )
    return model_root(Path(path))


def promote(new: ModelVersion, previous: ModelVersion | None) -> None:
    """Tags the new version first and untags the previous one afterwards: there is never a moment without a champion."""
    with tracking.azureml():
        client = MlflowClient()
        retry(lambda: client.set_model_version_tag(MODEL_NAME, new.version, CHAMPION_TAG, "true"), "tag champion")
        if previous is not None and previous.version != new.version:
            retry(lambda: client.delete_model_version_tag(MODEL_NAME, previous.version, CHAMPION_TAG), "untag champion")


def model_root(path: Path) -> Path:
    """Folder that contains model_config.json (the train output puts it in <output>/model)."""
    if (path / "model_config.json").exists():
        return path
    found = next(path.rglob("model_config.json"), None)
    if found is None:
        raise FileNotFoundError(f"no model_config.json in {path}")
    return found.parent
