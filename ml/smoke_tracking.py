"""ParkWatch — smoke test of the tracking switch inside an Azure ML job (aml/jobs/smoke-tracking.yml).

Checks, in the order train/promote will need them:
  1. the job sees its Azure ML MLflow (MLFLOW_TRACKING_URI, MLFLOW_RUN_ID)
  2. use_self_hosted(): the `pipeline` password from the Key Vault (cluster identity), the
     self-hosted server answers, and a run with params, a metric curve and an artifact is written
  3. the pipeline user can read the training experiment (epochs_from_last_cv searches it)
  4. inside azureml(): no basic-auth credentials, the `parknet` registry and the champion are
     reachable, the champion downloads, and an artifact can be uploaded to the job's own run
     (what register() will do)
  5. after azureml(): back on the self-hosted server, logging still works

The smoke run goes to its own experiment (default parkwatch-smoke): the pipeline user creates it
and becomes its owner, so it can be deleted from the UI without touching parkwatch-training.
Exits with an error at the first failed check.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient

import tracking
from registry import download, find_champion


def check(ok: bool, what: str) -> None:
    print(f"[{'OK' if ok else 'FAIL'}] {what}")
    if not ok:
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mlflow-url", required=True)
    p.add_argument("--keyvault-url", required=True)
    p.add_argument("--experiment", default="parkwatch-smoke")
    p.add_argument("--training-experiment", default="parkwatch-training")
    args = p.parse_args()

    # 1. Azure ML context of the job
    check(tracking.AZUREML_URI.startswith("azureml://"), f"Azure ML tracking URI ({tracking.AZUREML_URI.split('?')[0][:60]}...)")
    check(bool(tracking.AZUREML_RUN_ID), f"Azure ML job run id ({tracking.AZUREML_RUN_ID})")

    # 2. switch to the self-hosted server and write a run
    check(tracking.use_self_hosted(args.mlflow_url, args.keyvault_url, args.experiment), "use_self_hosted()")
    check("MLFLOW_RUN_ID" not in os.environ, "MLFLOW_RUN_ID removed")
    params = {"width": 1.0, "input_size": 96}
    with mlflow.start_run(run_name="smoke-" + tracking.run_name(params)) as run:
        mlflow.set_tag(tracking.TAG_AZURE_TRAIN_RUN, tracking.AZUREML_RUN_ID)
        mlflow.log_params(params)
        for step in range(5):
            mlflow.log_metric("smoke_curve", step * 0.1, step=step)
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "hello.txt"
            f.write_text("self-hosted artifact\n", encoding="utf-8")
            mlflow.log_artifact(str(f), "smoke")
        client = MlflowClient()
        logged = client.get_run(run.info.run_id)
        check(len(client.get_metric_history(run.info.run_id, "smoke_curve")) == 5, "self-hosted run: metric curve")
        check([a.path for a in client.list_artifacts(run.info.run_id, "smoke")] == ["smoke/hello.txt"],
              "self-hosted run: artifact (proxied to Blob)")
        check(logged.data.tags.get(tracking.TAG_AZURE_TRAIN_RUN) == tracking.AZUREML_RUN_ID, "self-hosted run: tags")

        # 3. read access to the training experiment
        training = client.get_experiment_by_name(args.training_experiment)
        check(training is not None, f"experiment {args.training_experiment} visible")
        runs = client.search_runs([training.experiment_id], max_results=50)
        check(True, f"{args.training_experiment}: {len(runs)} runs readable")

        # 4. Azure ML registry from inside the same process
        with tracking.azureml():
            check(mlflow.get_tracking_uri() == tracking.AZUREML_URI, "azureml(): global URI on the workspace")
            check("MLFLOW_TRACKING_PASSWORD" not in os.environ, "azureml(): basic-auth credentials hidden")
            champion = find_champion()
            check(champion is not None, f"champion found (parknet:{champion.version if champion else '-'})")
            with tempfile.TemporaryDirectory() as tmp:
                model = download(champion, Path(tmp))
                check((model / "model_config.json").exists(), f"champion downloaded ({sorted(p.name for p in model.iterdir())})")
                f = Path(tmp) / "smoke_azureml.txt"
                f.write_text("artifact on the Azure ML job run\n", encoding="utf-8")
                MlflowClient().log_artifact(tracking.AZUREML_RUN_ID, str(f), "smoke")
            check(True, "artifact uploaded to the Azure ML job run")

        # 5. back on the self-hosted server
        check(mlflow.get_tracking_uri() == args.mlflow_url.rstrip("/"), "after azureml(): global URI back")
        check(os.environ.get("MLFLOW_TRACKING_USERNAME") == tracking.PIPELINE_USER, "after azureml(): credentials back")
        mlflow.log_metric("after_azureml", 1.0)
        check("after_azureml" in MlflowClient().get_run(run.info.run_id).data.metrics, "self-hosted logging after azureml()")

    print(f"All checks passed. Run {run.info.run_id} in {args.mlflow_url.rstrip('/')} (experiment {args.experiment})")


if __name__ == "__main__":
    main()
