"""ParkWatch — copies past trainings from the Azure ML MLflow into the self-hosted MLflow.

One run per training, the same layout the pipeline writes from now on (ml/tracking.py):
  - params and metrics (full history: per-epoch curves) of the `train` step
  - params of the `promote` step prefixed with "promote/", its metrics (new/*, champion/*) as is
  - tags parknet_version, promotion (decision) and the source Azure ML run ids
  - artifacts eval/* (figures) and dataset/* (the model weights stay in the Azure ML registry)

train and promote are paired through the `parknet` registry: every version points to the
promote run that registered it and carries the train run id in the train_run_id tag. Trainings
that never reached promote are copied alone. Runs already copied are skipped (source tag), so
it can be re-run at any time.

Usage (repo root, az login; credentials of the self-hosted MLflow in environment variables):
  $env:MLFLOW_TRACKING_USERNAME = "pipeline"; $env:MLFLOW_TRACKING_PASSWORD = ...
  uv run --no-project --with-requirements tools/requirements-mlflow.txt python tools/import_mlflow_history.py --url https://<app> --dry-run
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import mlflow
from mlflow import MlflowClient
from mlflow.entities import Metric, Param

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from tracking import (  # noqa: E402
    PROMOTE_PARAM_PREFIX,
    TAG_AZURE_PROMOTE_RUN,
    TAG_AZURE_TRAIN_RUN,
    TAG_DECISION,
    TAG_VERSION,
    run_name,
)

SUBSCRIPTION = "48ee8f92-9a0c-4ecd-b6ce-83845f85bf75"
RESOURCE_GROUP = "parkwatch"
WORKSPACE = "parkwatch-workspace"
MODEL_NAME = "parknet"
ARTIFACT_DIRS = ("eval", "dataset")
BATCH = 900  # MLflow log_batch: at most 1000 metrics per call
# promote.py decisions before the repo was translated to English -> current values
LEGACY_DECISIONS = {
    "promovido": "promoted",
    "promovido_sin_champion": "promoted_no_champion",
    "descartado": "rejected",
    "sin_holdout": "no_holdout",
}


def all_runs(client: MlflowClient, experiment_id: str) -> list:
    runs, token = [], None
    while True:
        page = client.search_runs([experiment_id], max_results=500, page_token=token)
        runs.extend(page)
        token = page.token
        if not token:
            return runs


def metric_history(client: MlflowClient, run) -> list[Metric]:
    out = []
    for key in run.data.metrics:
        out.extend(Metric(key, m.value, m.timestamp, m.step) for m in client.get_metric_history(run.info.run_id, key))
    return out


@contextmanager
def global_uri(uri: str):
    """Artifact repositories resolve their server from the GLOBAL tracking URI, not from the
    client's: the Azure ML one takes its token from it, mlflow-artifacts:/ its host."""
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    try:
        yield
    finally:
        mlflow.set_tracking_uri(previous)


def copy_artifacts(src: MlflowClient, dst: MlflowClient, src_run: str, dst_run: str) -> int:
    n = 0
    with tempfile.TemporaryDirectory() as tmp:
        for d in ARTIFACT_DIRS:
            with global_uri(src.tracking_uri):
                if not src.list_artifacts(src_run, d):
                    continue
                local = src.download_artifacts(src_run, d, tmp)
            with global_uri(dst.tracking_uri):
                dst.log_artifacts(dst_run, local, d)
            n += sum(1 for p in Path(local).rglob("*") if p.is_file())
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="self-hosted MLflow, https://<FQDN of the Container App>")
    p.add_argument("--experiment", default="parkwatch-training", help="same name on both sides")
    p.add_argument("--dry-run", action="store_true", help="only list what would be copied")
    args = p.parse_args()

    ml = MLClient(DefaultAzureCredential(), SUBSCRIPTION, RESOURCE_GROUP, WORKSPACE)
    azure_uri = ml.workspaces.get(WORKSPACE).mlflow_tracking_uri
    src = MlflowClient(tracking_uri=azure_uri, registry_uri=azure_uri)
    dst = MlflowClient(tracking_uri=args.url)

    src_exp = src.get_experiment_by_name(args.experiment)
    if src_exp is None:
        raise SystemExit(f"Azure ML has no experiment {args.experiment}")
    runs = {r.info.run_id: r for r in all_runs(src, src_exp.experiment_id)}
    trains = sorted((r for r in runs.values() if r.data.params.get("arch") == "parknet" and r.info.status == "FINISHED"),
                    key=lambda r: r.info.start_time)
    promote_of = {}  # train run id -> (promote run, model version)
    for v in src.search_model_versions(f"name='{MODEL_NAME}'"):
        train_id = v.tags.get("train_run_id")
        if train_id and v.run_id in runs:
            promote_of[train_id] = (runs[v.run_id], v.version)

    dst_exp = dst.get_experiment_by_name(args.experiment)
    if dst_exp is None:
        raise SystemExit(f"{args.url} has no experiment {args.experiment} (mlflow_server/create_pipeline_user.py)")
    copied = {r.data.tags.get(TAG_AZURE_TRAIN_RUN): r for r in all_runs(dst, dst_exp.experiment_id)}
    done = {tid for tid, r in copied.items() if r.info.status == "FINISHED"}

    print(f"Azure ML {args.experiment}: {len(trains)} finished trainings, {len(promote_of)} with a registered model")
    for train in trains:
        tid = train.info.run_id
        promote, version = promote_of.get(tid, (None, None))
        name = run_name(train.data.params, train.info.start_time)
        decision = promote.data.params.get("decision", "?") if promote else "-"
        decision = LEGACY_DECISIONS.get(decision, decision)
        state = "already copied" if tid in done else ("would copy" if args.dry_run else "copying")
        print(f"  {name:22} parknet:{version or '-':<3} {decision:21} {state}")
        if tid in done or args.dry_run:
            continue

        tags = {TAG_AZURE_TRAIN_RUN: tid, "imported_from": "azureml"}
        if promote:
            tags |= {TAG_AZURE_PROMOTE_RUN: promote.info.run_id, TAG_VERSION: str(version), TAG_DECISION: decision}
        # A copy interrupted halfway (status RUNNING) is completed instead of duplicated:
        # what it already has is not logged again (a metric point would be duplicated).
        partial = copied.get(tid)
        if partial is None:
            partial = dst.create_run(dst_exp.experiment_id, start_time=train.info.start_time, tags=tags, run_name=name)
        rid = partial.info.run_id
        params = [Param(k, v) for k, v in train.data.params.items()]
        metrics = metric_history(src, train)
        if promote:
            params += [Param(PROMOTE_PARAM_PREFIX + k, decision if k == "decision" else v)
                       for k, v in promote.data.params.items()]
            metrics += metric_history(src, promote)
        if not partial.data.params:
            dst.log_batch(rid, params=params)
        if not partial.data.metrics:
            for i in range(0, len(metrics), BATCH):
                dst.log_batch(rid, metrics=metrics[i:i + BATCH])
        n_files = copy_artifacts(src, dst, tid, rid)
        end = (promote or train).info.end_time or train.info.end_time
        dst.set_terminated(rid, "FINISHED", end_time=end)
        print(f"    -> {len(params)} params, {len(metrics)} metric points, {n_files} files")

    if args.dry_run:
        print("Dry run: nothing written.")
    print(f"Done ({datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC)")


if __name__ == "__main__":
    main()
