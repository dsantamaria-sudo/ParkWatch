"""ParkWatch — evaluate & promote (last step of the training pipeline).

  new model (train output) + dataset (build_dataset output) + champion from the registry
    1. Registers the new model in `parknet` with its tags (every model is registered).
    2. Evaluates the new model and the champion on the SAME holdout of the current dataset,
       each with its own crop and normalisation (model_config.json).
    3. If there is no champion, or the new macro F1 is >= the champion's, it is promoted.

Without holdout days there is no promotion (plan.md, block 1): the model is only registered.

Tracking: if train logged to the self-hosted MLflow (model_config.json), promote completes that
same run (params prefixed "promote/", metrics new/* and champion/*, tags parknet_version and
promotion), so one run holds the whole training. Otherwise, or if the server cannot be reached,
it logs to its own Azure ML job run, as before. The model registry is always Azure ML.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

from data import CLASSES, SpotCrops, read_manifest
from inference import ParkNetPredictor
from metrics import evaluate
from registry import MODEL_NAME, download, find_champion, model_root, promote, register
import tracking


@torch.no_grad()
def holdout_metrics(model_dir: Path, data_dir: Path, rows: list[dict], args) -> dict:
    predictor = ParkNetPredictor(model_dir)
    c = predictor.config
    ds = SpotCrops(data_dir, rows, c["input_size"], c["crop_margin"], c["norm_mean"], c["norm_std"])
    ys, ps = [], []
    for x, y, _ in DataLoader(ds, batch_size=256, num_workers=args.num_workers):
        ps.append(predictor.model(x).argmax(1))
        ys.append(y)
    return evaluate(torch.cat(ys).numpy(), torch.cat(ps).numpy(), rows, CLASSES, args.dark_thr, args.low_contrast_thr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="train output")
    p.add_argument("--data-dir", type=Path, required=True, help="build_dataset output")
    p.add_argument("--holdout-days", nargs="*", default=[])
    p.add_argument("--dark-thr", type=float, default=50.0)
    p.add_argument("--low-contrast-thr", type=float, default=20.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--mlflow-url", default=None, help="self-hosted MLflow (the one train used)")
    p.add_argument("--keyvault-url", default=None, help="Key Vault with the pipeline user's password")
    args = p.parse_args()

    new_dir = model_root(args.model)
    config = json.loads((new_dir / "model_config.json").read_text(encoding="utf-8"))
    rows = read_manifest(args.data_dir)
    holdout = [r for r in rows if r["day"] in set(args.holdout_days)]

    # Complete the training run on the self-hosted MLflow if that is where train logged it.
    train_tracking = config.get("mlflow_tracking", "default")
    resume_id = None
    if args.mlflow_url and train_tracking == args.mlflow_url.rstrip("/"):
        if tracking.use_self_hosted(args.mlflow_url, args.keyvault_url):
            resume_id = config["mlflow_run_id"]
    elif args.mlflow_url:
        print(f"Tracking: train logged to {train_tracking}, not to {args.mlflow_url}: promote uses its own run")
    prefix = tracking.PROMOTE_PARAM_PREFIX if resume_id else ""

    with mlflow.start_run(run_id=resume_id) as run:
        if resume_id:
            print(f"Completing the training run {resume_id}")
            if tracking.AZUREML_RUN_ID:
                mlflow.set_tag(tracking.TAG_AZURE_PROMOTE_RUN, tracking.AZUREML_RUN_ID)
            tracking.link_azureml_run(run)

        new_m = holdout_metrics(new_dir, args.data_dir, holdout, args) if holdout else None
        azureml_train_run = config.get("azureml_run_id", "none")
        tags = {
            "holdout_days": ",".join(sorted(args.holdout_days)) or "none",
            "holdout_f1_macro": f"{new_m['f1_macro']:.4f}" if new_m else "none",
            # Azure ML run of the train step (the job page); before the self-hosted MLflow it
            # was also the run with the metrics.
            "train_run_id": azureml_train_run if azureml_train_run != "none" else config.get("mlflow_run_id", "none"),
            "dataset_fingerprint": config.get("dataset_fingerprint", "none"),
            "rois_version": config["rois_version"],
            "crop_margin": config["crop_margin"],
            "width": config["width"],
            "epochs": config["epochs"],
        }
        if train_tracking != "default":
            tags["self_hosted_run_id"] = config["mlflow_run_id"]
        new = register(new_dir, tags)
        mlflow.log_param(prefix + "new_version", new.version)
        mlflow.set_tag(tracking.TAG_VERSION, new.version)
        print(f"Registered {MODEL_NAME}:{new.version}")

        if not holdout:
            mlflow.log_param(prefix + "decision", "no_holdout")
            mlflow.set_tag(tracking.TAG_DECISION, "no_holdout")
            print("No holdout: no promotion (plan.md, block 1)")
            return
        mlflow.log_metrics({f"new/{k}": v for k, v in new_m.items() if isinstance(v, (int, float))})
        print(f"New {MODEL_NAME}:{new.version}: holdout macro F1 {new_m['f1_macro']:.4f} ({len(holdout)} crops)")

        champion = find_champion()
        if champion is None:
            decision = "promoted_no_champion"
        else:
            with tempfile.TemporaryDirectory() as tmp:
                champ_m = holdout_metrics(download(champion, Path(tmp)), args.data_dir, holdout, args)
            mlflow.log_param(prefix + "champion_version", champion.version)
            mlflow.log_metrics({f"champion/{k}": v for k, v in champ_m.items() if isinstance(v, (int, float))})
            print(f"Champion {MODEL_NAME}:{champion.version}: holdout macro F1 {champ_m['f1_macro']:.4f}")
            decision = "promoted" if new_m["f1_macro"] >= champ_m["f1_macro"] else "rejected"

        if decision.startswith("promoted"):
            promote(new, champion)
        mlflow.log_param(prefix + "decision", decision)
        mlflow.set_tag(tracking.TAG_DECISION, decision)
        print(f"Decision: {decision} -> champion = {MODEL_NAME}:{new.version if decision.startswith('promoted') else champion.version}")


if __name__ == "__main__":
    main()
