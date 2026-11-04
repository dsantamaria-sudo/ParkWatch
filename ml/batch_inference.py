"""ParkWatch — batch inference with the champion (first step of the inference pipeline).

  champion (parknet registry) + Blob photos of the last --last-days UTC days (or --days)
  + alignment table (align_photos): every photo is straightened before cropping
    -> <out>/predictions.json   {blob: {spot: {"label", "score", "probs"}}} + model version
                                + warps {blob: drift correction} (to move the polygons in Label Studio)

Same crop and preprocessing as in training (ParkNetPredictor + model_config.json).
"""
from __future__ import annotations

import argparse
import io
import json
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import mlflow
from azure.storage.blob import BlobServiceClient
from mlflow import MlflowClient
from PIL import Image

from alignment import Alignment
from azure_utils import credential, retry
from inference import ParkNetPredictor
from registry import MODEL_NAME, download, find_champion


def window(last_days: int, days: list[str]) -> list[date]:
    if days:
        return [date.fromisoformat(d) for d in days]
    today = datetime.now(timezone.utc).date()
    return [today - timedelta(days=k) for k in range(last_days, 0, -1)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rois", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--alignment", type=Path, default=None, help="align_photos output (alignment.csv)")
    p.add_argument("--last-days", type=int, default=3, help="UTC days before today")
    p.add_argument("--days", nargs="*", default=[], help="specific YYYY-MM-DD days (replace --last-days)")
    p.add_argument("--storage-account", default="parkwatch")
    p.add_argument("--container", default="raw")
    args = p.parse_args()

    rois = json.loads(args.rois.read_text(encoding="utf-8"))
    alignment = Alignment.from_dir(args.alignment)
    champion = find_champion(MlflowClient())
    if champion is None:
        raise SystemExit(f"No champion in {MODEL_NAME}: nothing to predict")
    container = BlobServiceClient(
        f"https://{args.storage_account}.blob.core.windows.net", credential=credential()
    ).get_container_client(args.container)

    with mlflow.start_run(), tempfile.TemporaryDirectory() as tmp:
        predictor = ParkNetPredictor(download(champion, Path(tmp)))
        classes = predictor.config["classes"]
        mlflow.log_params({"model": f"{MODEL_NAME}:{champion.version}", "rois_version": rois["version"]})
        predictions: dict[str, dict] = {}
        warps: dict[str, list[float]] = {}
        for day in window(args.last_days, args.days):
            blobs = retry(lambda: sorted(b.name for b in container.list_blobs(name_starts_with=f"{day:%Y/%m/%d}/")
                                         if b.name.endswith(".jpg")), f"list {day}")
            for blob in blobs:
                data = retry(lambda: container.download_blob(blob).readall(), blob)
                warps[blob] = list(alignment.warp(blob))
                probs = predictor.predict(Image.open(io.BytesIO(data)), rois, tuple(warps[blob]))
                predictions[blob] = {
                    spot: {"label": max(pr, key=pr.get), "score": round(max(pr.values()), 4),
                           "probs": {c: round(pr[c], 4) for c in classes}}
                    for spot, pr in probs.items()
                }
            print(f"{day}: {len(blobs)} photos")

        args.out.mkdir(parents=True, exist_ok=True)
        out = {"model": f"{MODEL_NAME}:{champion.version}", "rois_version": rois["version"],
               "predictions": predictions, "warps": warps}
        (args.out / "predictions.json").write_text(json.dumps(out), encoding="utf-8")
        n_free = sum(s["label"] == "free" for ph in predictions.values() for s in ph.values())
        mlflow.log_metrics({"n_photos": len(predictions), "n_free": n_free})
        print(f"{len(predictions)} photos predicted with {MODEL_NAME}:{champion.version} ({n_free} free spots)")


if __name__ == "__main__":
    main()
