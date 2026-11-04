"""ParkWatch — credentials, secrets and retries shared by the Azure ML jobs.

Credential:
  - In Azure ML: the cluster identity (parkwatch-identity), with AZURE_CLIENT_ID = its client id.
  - Locally: DefaultAzureCredential (az login).
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def credential():
    from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

    client_id = os.getenv("AZURE_CLIENT_ID")
    if client_id:
        return ManagedIdentityCredential(client_id=client_id)
    return DefaultAzureCredential()


def retry(fn: Callable[[], T], what: str, attempts: int = 3, base_delay: float = 10.0) -> T:
    """Retries network calls with growing waits (10 s, 20 s...), e.g. Label Studio waking up."""
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - any network failure counts
            status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            client_error = isinstance(status, int) and 400 <= status < 500 and status != 429
            if i == attempts or client_error:  # retrying does not fix a 4xx
                raise
            delay = base_delay * 2 ** (i - 1)
            print(f"  {what}: failure {i}/{attempts} ({type(exc).__name__}: {exc}); retrying in {delay:.0f} s")
            time.sleep(delay)
    raise AssertionError("unreachable")


def get_secret(vault_url: str, name: str) -> str:
    from azure.keyvault.secrets import SecretClient

    client = SecretClient(vault_url, credential())
    return retry(lambda: client.get_secret(name).value, f"Key Vault {name}")


def write_control_outputs(values: dict) -> None:
    """Control outputs (e.g. changed: bool) read by a conditional node of the pipeline.

    Azure ML reads them from the run property `azureml.pipeline.control` (JSON); it is written
    through the run history API with the job's MLflow credentials, as mldesigner does.
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.utils.rest_utils import http_request

    run = mlflow.active_run() or mlflow.start_run()
    creds = MlflowClient()._tracking_client.store.get_host_creds()  # noqa: SLF001
    creds.host = creds.host.replace("mlflow/v2.0", "mlflow/v1.0").replace("mlflow/v1.0", "history/v1.0")
    response = retry(
        lambda: http_request(
            host_creds=creds,
            endpoint=f"/experimentids/{run.info.experiment_id}/runs/{run.info.run_id}",
            method="PATCH",
            json={"runId": run.info.run_id, "properties": {"azureml.pipeline.control": json.dumps(values)}},
        ),
        "control outputs",
    )
    response.raise_for_status()
    print(f"Control outputs: {values}")


def latest_asset_file(asset_name: str, filename: str, max_versions: int = 20) -> tuple[bytes, str] | None:
    """(content, version) of a file from the newest version of a data asset that has it.

    Azure ML registers the output even when the job fails, so the latest version may be empty:
    versions are tried from newest to oldest. None if none of them has the file."""
    from azure.core.exceptions import ResourceNotFoundError
    from azure.storage.blob import BlobClient

    client = ml_client()
    # When the asset does not exist, data.get raises a generic MlException (not
    # ResourceNotFoundError) that retry would mistake for a network failure: check the list first.
    names = retry(lambda: {d.name for d in client.data.list()}, "list data assets")
    if asset_name not in names:
        return None
    versions = retry(lambda: list(client.data.list(name=asset_name)), "data asset versions")
    datastores: dict = {}
    for asset in sorted(versions, key=lambda a: int(a.version), reverse=True)[:max_versions]:
        # azureml://.../datastores/<datastore>/paths/<path>/
        datastore_name, rel = asset.path.split("/datastores/", 1)[1].split("/paths/", 1)
        if datastore_name not in datastores:
            datastores[datastore_name] = client.datastores.get(datastore_name)
        ds = datastores[datastore_name]
        blob = BlobClient(f"https://{ds.account_name}.blob.core.windows.net", ds.container_name,
                          f"{rel.rstrip('/')}/{filename}", credential=credential())
        try:
            return retry(lambda: blob.download_blob().readall(), f"{asset_name}:{asset.version}/{filename}"), asset.version
        except ResourceNotFoundError:
            print(f"  {asset_name}:{asset.version} has no {filename} (failed job?), trying the previous one")
    return None


def ml_client():
    """MLClient of the workspace the job runs in (AZUREML_ARM_* variables set by Azure ML)."""
    from azure.ai.ml import MLClient

    return MLClient(
        credential(),
        os.environ["AZUREML_ARM_SUBSCRIPTION"],
        os.environ["AZUREML_ARM_RESOURCEGROUP"],
        os.environ["AZUREML_ARM_WORKSPACE_NAME"],
    )
