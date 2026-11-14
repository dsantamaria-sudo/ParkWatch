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
  uv run python aml/pipelines/training.py --experiment crop0 --crop-margin 0.0   # experiment
  uv run python aml/pipelines/training.py --experiment no-new-days --final-epochs 37 --exclude-days 2026-11-09

--experiment NAME: only `train` (GPU), on the latest registered dataset (no rebuild), always with
cross-validation (or exactly --final-epochs), holdout kept out of training but NOT evaluated, no
registration/promotion.
Experiments are compared on the cross-validation in MLflow (tag `experiment` = NAME); the
holdout is only used once, to promote the winner.
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
LANDMARKS_PROJECT = 7  # parkwatch-landmarks: anchors of the camera landmarks (align_photos)
KEYVAULT_URL = "https://parkwatch-keyvault.vault.azure.net"
# Self-hosted MLflow (infra/mlflow.bicep): train and promote log the training run there.
MLFLOW_URL = "https://parkwatch-mlflow.proudcliff-6d0c960a.spaincentral.azurecontainerapps.io"
DATASET_ASSET = "parkwatch-crops"
ALIGNMENT_ASSET = "parkwatch-alignment"
# Fixed, whole days chosen by hand: a day that goes in never comes out.
# 23/10: the most varied day. 3/11: reserved before labelling (no model has seen it), with the
# camera moved, so that the holdout also measures robustness to drift. 11/11 and 12/11: added
# before any of their photos was labelled, so that one rare episode (the large van of 23/10 on
# p04/p05) weighs less in the promotion.
HOLDOUT_DAYS: list[str] = ["2026-10-23", "2026-11-03", "2026-11-11", "2026-11-12"]

# Production crop configuration (also the defaults of the nightly/weekly schedules and of the
# experiments). Changing it changes what the next scheduled run trains and promotes.
# JITTER 0.05 since 13/11: the alignment table tracked again (TRACKING_VERSION 2) gives more
# consistent crops, and without jitter the models trained on it generalised worse than the
# champion trained on the old, noisier table (holdout macro F1 0.966 vs 0.977). Same data with
# jitter 0.05: 0.978 (experiment keep10_jit05_v21).
CROP_MARGIN = 0.10
JITTER = 0.05

# Maximum time per step, in seconds.
TIMEOUT_DATASET = 60 * 60
TIMEOUT_ALIGN = 2 * 60 * 60  # the first run measures every photo since 16/10
TIMEOUT_TRAIN = 3 * 60 * 60
TIMEOUT_PROMOTE = 30 * 60

ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = ROOT / "aml" / "components"
ROIS = ROOT / "label_studio" / "rois.json"


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


def align_step(component, rois, client_id: str):
    align = component(rois=rois, ls_url=LS_URL, landmarks_project=LANDMARKS_PROJECT,
                      keyvault_url=KEYVAULT_URL, asset_name=ALIGNMENT_ASSET)
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
    def training(rois: Input, force: bool = False, run_cv: bool = False,
                 crop_margin: float = CROP_MARGIN, jitter: float = JITTER, width: float = 1.0):
        align = align_step(align_photos, rois, client_id)
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
            crop_margin=crop_margin, jitter=jitter, width=width,
            mlflow_url=MLFLOW_URL, keyvault_url=KEYVAULT_URL, **holdout,
        )
        fit.compute = "gpu-cluster"
        fit.environment_variables = env  # same identity on both clusters (Key Vault)
        fit.set_limits(timeout=TIMEOUT_TRAIN)

        promote = evaluate_promote(
            model=fit.outputs.model, dataset=build.outputs.dataset,
            mlflow_url=MLFLOW_URL, keyvault_url=KEYVAULT_URL, **holdout,
        )
        promote.environment_variables = env
        promote.set_limits(timeout=TIMEOUT_PROMOTE)

        condition(check.outputs.changed, true_block=[build, fit, promote])

    return training


def build_experiment(client_id: str, final_epochs: int | None = None, exclude_days: list[str] | None = None):
    """Pipeline of one experiment: train with CV on a fixed dataset, nothing registered.

    final_epochs: no CV, the final model trains exactly that many epochs (e.g. to reproduce a
    production run). exclude_days: days left out of training (e.g. to test whether they hurt)."""
    train = load_component(COMPONENTS / "train.yml")
    fixed = {"run_cv": False, "final_epochs": final_epochs} if final_epochs else {"run_cv": True}
    if exclude_days:
        fixed["exclude_days"] = " ".join(exclude_days)

    @pipeline(name="parkwatch_experiment", default_compute="gpu-cluster")
    def experiment(dataset: Input, note: str, crop_margin: float = CROP_MARGIN, jitter: float = JITTER,
                   width: float = 1.0, input_size: int = 96, epochs: int = 40):
        holdout = {"holdout_days": " ".join(HOLDOUT_DAYS)} if HOLDOUT_DAYS else {}
        fit = train(
            dataset=dataset, crop_margin=crop_margin, jitter=jitter, width=width, input_size=input_size,
            epochs=epochs, holdout_eval=False, note=note, **fixed,
            mlflow_url=MLFLOW_URL, keyvault_url=KEYVAULT_URL, **holdout,
        )
        fit.environment_variables = {"AZURE_CLIENT_ID": client_id, "PYTHONUNBUFFERED": "1"}
        fit.set_limits(timeout=TIMEOUT_TRAIN)

    return experiment


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="rebuild even if the fingerprint did not change")
    p.add_argument("--run-cv", action="store_true", help="full cross-validation (default: epochs of the last run with CV)")
    # Hyperparameters to compare in MLflow; the defaults are the production ones.
    p.add_argument("--crop-margin", type=float, default=CROP_MARGIN, help="crop margin around the polygon bbox")
    p.add_argument("--jitter", type=float, default=JITTER, help="+-fraction of crop shift/scale (training only)")
    p.add_argument("--width", type=float, default=1.0, help="ParkNet filter multiplier")
    p.add_argument("--input-size", type=int, default=96, help="input side in px (experiments only)")
    p.add_argument("--epochs", type=int, default=40, help="maximum epochs of the CV (experiments only)")
    p.add_argument("--experiment", metavar="NAME", help="experiment: train with CV on the latest dataset, no promotion")
    p.add_argument("--dataset-version", help="experiments: version of the dataset asset (default: the latest)")
    p.add_argument("--final-epochs", type=int, help="experiments: no CV, train the final model exactly N epochs")
    p.add_argument("--exclude-days", nargs="*", default=[], help="experiments: YYYY-MM-DD left out of training")
    args = p.parse_args()

    client = MLClient(DefaultAzureCredential(), SUBSCRIPTION, RESOURCE_GROUP, WORKSPACE)
    if args.experiment:
        # the SDK does not resolve name@latest for pipeline inputs: the version is looked up here
        version = args.dataset_version or client.data.get(DATASET_ASSET, label="latest").version
        print(f"Dataset {DATASET_ASSET}:{version}")
        job = build_experiment(cluster_client_id(client, "cpu-cluster"), args.final_epochs, args.exclude_days)(
            dataset=Input(type="uri_folder", path=f"azureml:{DATASET_ASSET}:{version}"),
            note=args.experiment, crop_margin=args.crop_margin, jitter=args.jitter,
            width=args.width, input_size=args.input_size, epochs=args.epochs,
        )
        job = client.jobs.create_or_update(configure(job, "parkwatch-experiments"))
        print(f"Experiment {args.experiment} submitted: {job.name}\n  {job.studio_url}")
        return

    training = build_pipeline(cluster_client_id(client, "cpu-cluster"))
    job = training(
        rois=Input(type="uri_file", path=str(ROIS)),
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
