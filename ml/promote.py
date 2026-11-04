"""ParkWatch — evaluate & promote (last step of the training pipeline).

  new model (train output) + dataset (build_dataset output) + champion from the registry
    1. Registers the new model in `parknet` with its tags (every model is registered).
    2. Evaluates the new model and the champion on the SAME holdout of the current dataset,
       each with its own crop and normalisation (model_config.json).
    3. If there is no champion, or the new macro F1 is >= the champion's, it is promoted.

Without holdout days there is no promotion (plan.md, block 1): the model is only registered.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import torch
from mlflow import MlflowClient
from torch.utils.data import DataLoader

from data import CLASSES, SpotCrops, read_manifest
from inference import ParkNetPredictor
from metrics import evaluate
from registry import MODEL_NAME, download, find_champion, model_root, promote, register


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
    args = p.parse_args()

    new_dir = model_root(args.model)
    config = json.loads((new_dir / "model_config.json").read_text(encoding="utf-8"))
    rows = read_manifest(args.data_dir)
    holdout = [r for r in rows if r["day"] in set(args.holdout_days)]
    client = MlflowClient()

    with mlflow.start_run():
        new_m = holdout_metrics(new_dir, args.data_dir, holdout, args) if holdout else None
        tags = {
            "holdout_days": ",".join(sorted(args.holdout_days)) or "none",
            "holdout_f1_macro": f"{new_m['f1_macro']:.4f}" if new_m else "none",
            "train_run_id": config.get("mlflow_run_id", "none"),
            "dataset_fingerprint": config.get("dataset_fingerprint", "none"),
            "rois_version": config["rois_version"],
            "crop_margin": config["crop_margin"],
            "width": config["width"],
            "epochs": config["epochs"],
        }
        new = register(new_dir, tags)
        mlflow.log_param("new_version", new.version)
        print(f"Registered {MODEL_NAME}:{new.version}")

        if not holdout:
            mlflow.log_param("decision", "no_holdout")
            print("No holdout: no promotion (plan.md, block 1)")
            return
        mlflow.log_metrics({f"new/{k}": v for k, v in new_m.items() if isinstance(v, (int, float))})
        print(f"New {MODEL_NAME}:{new.version}: holdout macro F1 {new_m['f1_macro']:.4f} ({len(holdout)} crops)")

        champion = find_champion(client)
        if champion is None:
            decision = "promoted_no_champion"
        else:
            with tempfile.TemporaryDirectory() as tmp:
                champ_m = holdout_metrics(download(champion, Path(tmp)), args.data_dir, holdout, args)
            mlflow.log_param("champion_version", champion.version)
            mlflow.log_metrics({f"champion/{k}": v for k, v in champ_m.items() if isinstance(v, (int, float))})
            print(f"Champion {MODEL_NAME}:{champion.version}: holdout macro F1 {champ_m['f1_macro']:.4f}")
            decision = "promoted" if new_m["f1_macro"] >= champ_m["f1_macro"] else "rejected"

        if decision.startswith("promoted"):
            promote(client, new, champion)
        mlflow.log_param("decision", decision)
        print(f"Decision: {decision} -> champion = {MODEL_NAME}:{new.version if decision.startswith('promoted') else champion.version}")


if __name__ == "__main__":
    main()
