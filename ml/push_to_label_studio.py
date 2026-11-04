"""ParkWatch — brings the champion's predictions to Label Studio (second step of inference).

For every photo in predictions.json:
  - annotated task (also cancelled/skipped): left untouched
  - unannotated task: the labels of its pre-annotation are replaced by the model's, in the
    same prediction (same model_version; nothing is added and the project is not changed), and
    each polygon is placed over its real spot in that photo (rois.json ROIs moved with the
    photo's drift correction): the reviewer sees the spots where they are even if the camera moved
  - no task: only with --create-missing (the pipeline creates the tasks since it took over from
    the Pi), with one region per spot (rois.json id) and the model's labels
Each region carries its probability as score; the prediction carries the photo's lowest
confidence (to sort in Label Studio and review the doubtful ones first).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from alignment import IDENTITY, transform_points
from azure_utils import get_secret
from ls_client import (FROM_NAME, get_prediction, import_tasks, is_annotated, list_tasks, ls_headers,
                       update_prediction)

TO_NAME = "image"  # label_studio/labeling_config.xml


def photo_polygons(rois: dict, warp) -> dict[str, list[list[float]]]:
    """rois.json ROIs (reference geometry) where they fall in this photo; always computed from
    rois.json, so a task updated on several nights does not accumulate corrections."""
    return {s["id"]: transform_points(s["points"], tuple(warp), tuple(rois["image_size"])) for s in rois["spots"]}


def relabel(result: list[dict], spots: dict[str, dict], points: dict[str, list]) -> list[dict]:
    """Same list of regions with the model's label, score and position for the predicted spots."""
    out = []
    for region in result:
        rid = region.get("id")
        pred = spots.get(rid) if region.get("from_name") == FROM_NAME else None
        if pred is not None:
            value = {**region["value"], "polygonlabels": [pred["label"]]}
            if rid in points:
                value["points"] = points[rid]
            region = {**region, "score": pred["score"], "value": value}
        out.append(region)
    return out


def new_task(blob: str, container: str, rois: dict, spots: dict[str, dict], points: dict[str, list]) -> dict:
    result = [
        {"id": s["id"], "from_name": FROM_NAME, "to_name": TO_NAME, "type": "polygonlabels",
         "score": spots[s["id"]]["score"],
         "value": {"points": points[s["id"]], "closed": True, "polygonlabels": [spots[s["id"]]["label"]]}}
        for s in rois["spots"]
    ]
    return {
        "data": {"image": f"azure-blob://{container}/{blob}", "blob": blob},
        "predictions": [{"model_version": f"rois-{rois['version']}",
                         "score": min(r["score"] for r in result), "result": result}],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True, help="batch_inference output")
    p.add_argument("--rois", type=Path, required=True)
    p.add_argument("--project", type=int, required=True)
    p.add_argument("--ls-url", required=True)
    p.add_argument("--keyvault-url", required=True)
    p.add_argument("--secret-name", default="label-studio-token")
    p.add_argument("--container", default="raw")
    p.add_argument("--create-missing", default="false", help="true = create tasks for photos without one")
    args = p.parse_args()

    pred = json.loads((args.predictions / "predictions.json").read_text(encoding="utf-8"))
    rois = json.loads(args.rois.read_text(encoding="utf-8"))
    ls_url = args.ls_url.rstrip("/")
    headers = ls_headers(ls_url, os.getenv("LS_TOKEN") or get_secret(args.keyvault_url, args.secret_name))

    tasks = {t["data"].get("blob"): t for t in list_tasks(ls_url, headers, args.project) if t.get("data", {}).get("blob")}
    project_version = f"rois-{rois['version']}"
    counts = {"updated": 0, "annotated_untouched": 0, "no_prediction": 0, "no_task": 0, "created": 0}
    missing = []
    for blob, spots in pred["predictions"].items():
        points = photo_polygons(rois, pred.get("warps", {}).get(blob, IDENTITY))
        task = tasks.get(blob)
        if task is None:
            counts["no_task"] += 1
            missing.append(new_task(blob, args.container, rois, spots, points))
            continue
        if is_annotated(task):
            counts["annotated_untouched"] += 1
            continue
        preds = [get_prediction(ls_url, headers, x) for x in task.get("predictions", [])]
        if not preds:
            counts["no_prediction"] += 1
            continue
        # the one Label Studio shows (project model_version); otherwise the newest
        target = next((x for x in preds if x.get("model_version") == project_version), max(preds, key=lambda x: x["id"]))
        update_prediction(ls_url, headers, target["id"], relabel(target["result"], spots, points),
                          min(s["score"] for s in spots.values()))
        counts["updated"] += 1

    if missing and args.create_missing.lower() == "true":
        import_tasks(ls_url, headers, args.project, missing)
        counts["created"] = len(missing)
    print(f"Predictions of {pred['model']} in project {args.project}: {counts}")


if __name__ == "__main__":
    main()
