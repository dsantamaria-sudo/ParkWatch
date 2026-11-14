"""ParkWatch — experiment tracking on the self-hosted MLflow server.

train and promote log params, metrics and figures to the self-hosted MLflow (infra/mlflow.bicep),
one run per training: train creates it, promote adds the comparison with the champion and the
decision. Everything else stays on the Azure ML MLflow: the pipeline control outputs, the small
data/inference steps and the `parknet` model registry (registry.py), which the jobs reach
through `azureml()`.

Inside an Azure ML job MLflow points to the workspace (MLFLOW_TRACKING_URI) and resumes the job's
own run (MLFLOW_RUN_ID). use_self_hosted() keeps both values (AZUREML_URI, AZUREML_RUN_ID) and
switches the global MLflow to the self-hosted server. If the server cannot be reached the job
stays on Azure ML: the training never fails because of the tracking server.

This module also fixes the run layout (run names and the tags that pair train and promote).
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone

# Captured on import, before use_self_hosted() replaces them.
AZUREML_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
AZUREML_RUN_ID = os.environ.get("MLFLOW_RUN_ID")

PIPELINE_USER = "pipeline"
PASSWORD_SECRET = "mlflow-pipeline-password"
PROMOTE_PARAM_PREFIX = "promote/"
TAG_AZURE_TRAIN_RUN = "azureml_train_run"
TAG_AZURE_PROMOTE_RUN = "azureml_promote_run"
TAG_VERSION = "parknet_version"
TAG_DECISION = "promotion"
TAG_SELF_HOSTED_RUN = "self_hosted_run"  # on the Azure ML job run: link to the self-hosted run


def run_name(params: dict, start_time_ms: int | None = None) -> str:
    """<UTC date of the run>_w<width>_<input_size>px, e.g. 2026-11-04_w1.0_96px."""
    start = datetime.fromtimestamp((start_time_ms or time.time() * 1000) / 1000, timezone.utc)
    return f"{start:%Y-%m-%d}_w{params.get('width', '?')}_{params.get('input_size', '?')}px"


def _wait_until_up(url: str, timeout_s: int = 300) -> bool:
    """The Container App scales to zero: a cold start takes ~30 s."""
    import requests

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(f"{url}/health", timeout=30).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(10)
    return False


def use_self_hosted(url: str | None, keyvault_url: str | None, experiment: str | None = None) -> bool:
    """Points the global MLflow to the self-hosted server (and to `experiment`, if given: promote
    resumes an existing run instead). False (and Azure ML unchanged) if it is not configured or
    cannot be reached."""
    import mlflow

    if not url:
        print("Tracking: Azure ML (no --mlflow-url)")
        return False
    url = url.rstrip("/")
    try:
        password = os.getenv("MLFLOW_PIPELINE_PASSWORD")
        if not password:
            from azure_utils import get_secret

            password = get_secret(keyvault_url, PASSWORD_SECRET)
        if not _wait_until_up(url):
            raise ConnectionError(f"{url}/health does not answer")
        os.environ["MLFLOW_TRACKING_USERNAME"] = PIPELINE_USER
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        # The job's run id belongs to Azure ML: with it set, start_run() would try to resume a
        # run that does not exist on the self-hosted server.
        os.environ.pop("MLFLOW_RUN_ID", None)
        os.environ["MLFLOW_TRACKING_URI"] = url
        mlflow.set_tracking_uri(url)
        if experiment:
            mlflow.set_experiment(experiment)
    except Exception as exc:  # noqa: BLE001 - any failure: keep tracking on Azure ML
        for key in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
            os.environ.pop(key, None)
        if AZUREML_RUN_ID:
            os.environ["MLFLOW_RUN_ID"] = AZUREML_RUN_ID
        if AZUREML_URI:
            os.environ["MLFLOW_TRACKING_URI"] = AZUREML_URI
            mlflow.set_tracking_uri(AZUREML_URI)
        print(f"WARNING tracking: self-hosted MLflow unavailable ({type(exc).__name__}: {exc}); using Azure ML")
        return False
    print(f"Tracking: {url} (experiment {experiment or '-'}, user {PIPELINE_USER})")
    return True


@contextmanager
def azureml():
    """Global MLflow temporarily on the Azure ML workspace (model registry). No-op outside a job
    or when the tracking was not switched. Fluent logging (mlflow.log_*) must not be called inside."""
    import mlflow

    previous = mlflow.get_tracking_uri()
    if not AZUREML_URI or previous == AZUREML_URI:
        yield
        return
    # The basic-auth credentials of the self-hosted server must not reach the Azure ML store.
    saved = {k: os.environ.pop(k) for k in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD") if k in os.environ}
    mlflow.set_tracking_uri(AZUREML_URI)
    mlflow.set_registry_uri(AZUREML_URI)
    try:
        yield
    finally:
        mlflow.set_tracking_uri(previous)
        mlflow.set_registry_uri(previous)
        os.environ.update(saved)


def link_azureml_run(run) -> None:
    """Tags the Azure ML job run with the URL of the self-hosted run (the job page in the studio
    shows no metrics otherwise). Best effort: a failure only prints a warning."""
    import mlflow
    from mlflow import MlflowClient

    if not AZUREML_RUN_ID or run.info.run_id == AZUREML_RUN_ID:
        return
    url = f"{mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
    try:
        with azureml():
            MlflowClient().set_tag(AZUREML_RUN_ID, TAG_SELF_HOSTED_RUN, url)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING tracking: could not tag the Azure ML run ({type(exc).__name__}: {exc})")
