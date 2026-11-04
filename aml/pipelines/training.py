"""ParkWatch — training pipeline on Azure ML.

  align_photos (CPU) -> parkwatch-alignment table ----------------------------> drift_alert (CPU)
    -> check_dataset (CPU) --changed--> [condition] --> build_dataset (CPU) -> data asset parkwatch-crops
                                                       -> train (GPU) -> evaluate & promote (CPU)

align_photos measures the camera drift on the new photos (always); check and build straighten
every photo with that table. drift_alert fails (and sends the email) if the camera moved too
much, without stopping the rest. If the dataset fingerprint did not change, the whole branch
is skipped. promote registers the model in parknet and tags it champion if its macro F1 on the
holdout (HOLDOUT_DAYS) is >= the current one's.

Usage (from the repo root, with az login):
  uv run python aml/pipelines/training.py                  # epochs of the last run with CV
  uv run python aml/pipelines/training.py --run-cv         # full cross-validation
  uv run python aml/pipelines/training.py --force          # rebuild even if the fingerprint did not change
  uv run python aml/pipelines/training.py --force --run-cv --crop-margin 0.0   # experiment
"""
# No `from __future__ import annotations`: @pipeline reads the annotations (Input, bool) as types.
import argparse
import os
from pathlib import Path

# The if_else node (dsl.condition) is only serialised with the SDK preview features enabled.
os.environ.setdefault("AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED", "true")

from azure.ai.ml import Input, MLClient, load_component  # noqa: E402
from azure.ai.ml.dsl import pipeline  # noqa: E402
from azure.ai.ml.dsl._condition import condition  # noqa: E402  (control flow, preview)
from azure.identity import DefaultAzureCredential  # noqa: E402

SUBSCRIPTION = "48ee8f92-9a0c-4ecd-b6ce-83845f85bf75"
RESOURCE_GROUP = "parkwatch"
WORKSPACE = "parkwatch-workspace"

LS_URL = "https://park-watch-label-studio.salmonbay-c1ca7758.spaincentral.azurecontainerapps.io"
LS_PROJECT = 5
KEYVAULT_URL = "https://parkwatch-keyvault.vault.azure.net"
DATASET_ASSET = "parkwatch-crops"
ALIGNMENT_ASSET = "parkwatch-alignment"
# Fixed, whole days chosen by hand (plan.md block 1): a day that goes in never comes out.
# 23/10: the most varied day. 3/11: reserved before labelling (no model has seen it), with the
# camera moved, so that the holdout also measures robustness to drift.
HOLDOUT_DAYS: list[str] = ["2026-10-23", "2026-11-03"]

# Maximum time per step (block 4), in seconds.
TIMEOUT_DATASET = 60 * 60
TIMEOUT_ALIGN = 2 * 60 * 60  # the first run measures every photo since 16/10
TIMEOUT_TRAIN = 3 * 60 * 60
TIMEOUT_PROMOTE = 30 * 60

ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = ROOT / "aml" / "components"
ROIS = ROOT / "label_studio" / "rois.json"
STATIC_MASK = ROOT / "label_studio" / "static_mask.json"


def cluster_client_id(client: MLClient, cluster: str) -> str:
    """Client id of the cluster's user-assigned identity (used by the scripts to reach Azure)."""
    identity = client.compute.get(cluster).identity
    return identity.user_assigned_identities[0].client_id


def configure(job, experiment: str):
    """Settings shared by manual and scheduled runs."""
    # Never reuse steps from previous runs: check_dataset and align_photos query external
    # state, and build_dataset only runs when the fingerprint changed.
    job.settings.force_rerun = True
    # drift_alert may fail on purpose (email) without stopping the rest of the pipeline.
    job.settings.continue_on_step_failure = True
    job.experiment_name = experiment
    return job


def align_step(component, rois, mask, client_id: str):
    align = component(rois=rois, mask=mask, asset_name=ALIGNMENT_ASSET)
    align.environment_variables = {"AZURE_CLIENT_ID": client_id, "PYTHONUNBUFFERED": "1"}
    align.set_limits(timeout=TIMEOUT_ALIGN)
    return align


def build_pipeline(client_id: str):
    align_photos = load_component(COMPONENTS / "align_photos.yml")
    drift_alert = load_component(COMPONENTS / "drift_alert.yml")
    check_dataset = load_component(COMPONENTS / "check_dataset.yml")
    build_dataset = load_component(COMPONENTS / "build_dataset.yml")
    train = load_component(COMPONENTS / "train.yml")
    evaluate_promote = load_component(COMPONENTS / "promote.yml")
    env = {"AZURE_CLIENT_ID": client_id, "PYTHONUNBUFFERED": "1"}

    @pipeline(name="parkwatch_training", default_compute="cpu-cluster")
    def training(rois: Input, mask: Input, force: bool = False, run_cv: bool = False,
                 crop_margin: float = 0.10, jitter: float = 0.0, width: float = 1.0):
        align = align_step(align_photos, rois, mask, client_id)
        align.outputs.alignment.name = ALIGNMENT_ASSET  # a new version of the table every night
        alert = drift_alert(alignment=align.outputs.alignment)
        alert.environment_variables = {"PYTHONUNBUFFERED": "1"}

        check = check_dataset(
            rois=rois, alignment=align.outputs.alignment, ls_url=LS_URL, ls_project=LS_PROJECT,
            keyvault_url=KEYVAULT_URL, asset_name=DATASET_ASSET, force=force,
        )
        check.environment_variables = env
        check.set_limits(timeout=TIMEOUT_DATASET)

        build = build_dataset(rois=rois, export=check.outputs.export, alignment=align.outputs.alignment)
        build.environment_variables = env
        build.set_limits(timeout=TIMEOUT_DATASET)
        build.outputs.dataset.name = DATASET_ASSET  # Azure ML registers a new version

        holdout = {"holdout_days": " ".join(HOLDOUT_DAYS)} if HOLDOUT_DAYS else {}
        fit = train(
            dataset=build.outputs.dataset, run_cv=run_cv,
            crop_margin=crop_margin, jitter=jitter, width=width, **holdout,
        )
        fit.compute = "gpu-cluster"
        fit.environment_variables = {"PYTHONUNBUFFERED": "1"}
        fit.set_limits(timeout=TIMEOUT_TRAIN)

        promote = evaluate_promote(model=fit.outputs.model, dataset=build.outputs.dataset, **holdout)
        promote.environment_variables = {"PYTHONUNBUFFERED": "1"}
        promote.set_limits(timeout=TIMEOUT_PROMOTE)

        condition(check.outputs.changed, true_block=[build, fit, promote])

    return training


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="rebuild even if the fingerprint did not change")
    p.add_argument("--run-cv", action="store_true", help="full cross-validation (default: epochs of the last run with CV)")
    # Hyperparameters to compare in MLflow (block 2); the defaults are those of the base model.
    p.add_argument("--crop-margin", type=float, default=0.10, help="crop margin around the polygon bbox")
    p.add_argument("--jitter", type=float, default=0.0, help="+-fraction of crop shift/scale (training only)")
    p.add_argument("--width", type=float, default=1.0, help="ParkNet filter multiplier")
    args = p.parse_args()

    client = MLClient(DefaultAzureCredential(), SUBSCRIPTION, RESOURCE_GROUP, WORKSPACE)
    training = build_pipeline(cluster_client_id(client, "cpu-cluster"))
    job = training(
        rois=Input(type="uri_file", path=str(ROIS)),
        mask=Input(type="uri_file", path=str(STATIC_MASK)),
        force=args.force,
        run_cv=args.run_cv,
        crop_margin=args.crop_margin,
        jitter=args.jitter,
        width=args.width,
    )
    job = client.jobs.create_or_update(configure(job, "parkwatch-training"))
    print(f"Pipeline submitted: {job.name}\n  {job.studio_url}")


if __name__ == "__main__":
    main()
