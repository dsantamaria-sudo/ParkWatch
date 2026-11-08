"""ParkWatch — inference pipeline on Azure ML (independent from the training one).

  align_photos (CPU) -> batch_inference (CPU, champion) -> push_to_label_studio (CPU)

align_photos completes the alignment table with any missing photos (without registering a new
version: the training pipeline does that) and batch_inference straightens every spot with it.

Unannotated tasks of the last 3 days (or of --days) get the champion's labels instead of the
"all occupied" pre-annotation, with each polygon placed over its real spot. Annotated tasks are
never touched. --create-missing creates the tasks of photos that do not have one yet.

Usage (from the repo root, with az login):
  uv run python aml/pipelines/inference.py
  uv run python aml/pipelines/inference.py --days 2026-10-31 2026-11-01
"""
# No `from __future__ import annotations`: @pipeline reads the annotations as types.
import argparse

from azure.ai.ml import Input, MLClient, load_component
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

from training import (COMPONENTS, KEYVAULT_URL, LS_PROJECT, LS_URL, RESOURCE_GROUP, ROIS, SUBSCRIPTION,
                      WORKSPACE, align_step, cluster_client_id, configure)

TIMEOUT_INFERENCE = 30 * 60  # block 4


def build_pipeline(client_id: str, days: list[str]):
    align_photos = load_component(COMPONENTS / "align_photos.yml")
    batch_inference = load_component(COMPONENTS / "batch_inference.yml")
    push_to_label_studio = load_component(COMPONENTS / "push_to_label_studio.yml")
    env = {"AZURE_CLIENT_ID": client_id, "PYTHONUNBUFFERED": "1"}
    window = {"days": " ".join(days)} if days else {}

    @pipeline(name="parkwatch_inference", default_compute="cpu-cluster")
    def inference(rois: Input, last_days: int = 3, create_missing: bool = False):
        align = align_step(align_photos, rois, client_id)
        predict = batch_inference(rois=rois, alignment=align.outputs.alignment, last_days=last_days, **window)
        predict.environment_variables = env
        predict.set_limits(timeout=TIMEOUT_INFERENCE)

        push = push_to_label_studio(
            predictions=predict.outputs.predictions, rois=rois,
            ls_url=LS_URL, ls_project=LS_PROJECT, keyvault_url=KEYVAULT_URL, create_missing=create_missing,
        )
        push.environment_variables = env
        push.set_limits(timeout=TIMEOUT_INFERENCE)

    return inference


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", nargs="*", default=[], help="specific YYYY-MM-DD days (default: the last 3 days)")
    p.add_argument("--create-missing", action="store_true", help="create tasks for photos without one")
    args = p.parse_args()

    client = MLClient(DefaultAzureCredential(), SUBSCRIPTION, RESOURCE_GROUP, WORKSPACE)
    inference = build_pipeline(cluster_client_id(client, "cpu-cluster"), args.days)
    job = inference(
        rois=Input(type="uri_file", path=str(ROIS)),
        create_missing=args.create_missing,
    )
    job = client.jobs.create_or_update(configure(job, "parkwatch-inference"))
    print(f"Pipeline submitted: {job.name}\n  {job.studio_url}")


if __name__ == "__main__":
    main()
