"""ParkWatch — decides whether the dataset has to be rebuilt (first step of the training pipeline).

  Label Studio (JSON export) + rois.json + alignment table (align_photos)
    -> <out>/export.json   full export, so build_dataset does not request it again
    -> <out>/check.json    new fingerprint, fingerprint of the latest data asset version, changed

The fingerprint (SHA-256) covers what defines the dataset: final labels (photo, spot, label,
including exclude), rois.json, the crop margins and the alignment of every labelled photo
(rounded to half a pixel). Annotation timestamps are left out: saving a task again without
changing its labels does not create a new version.

Usage (locally, with az login):
  python check_dataset.py --rois ../label_studio/rois.json --out ../data/check \
      --project 5 --ls-url https://... --keyvault-url https://parkwatch-keyvault.vault.azure.net
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from alignment import Alignment
from azure_utils import get_secret, latest_asset_file, write_control_outputs
from crops import STORE_MARGIN
from ls_client import export_tasks, latest_labels

FINGERPRINT_FORMAT = 2  # bump when what goes into the fingerprint changes (2: per-photo alignment)


def dataset_fingerprint(tasks: list[dict], rois: dict, quality_margin: float, alignment: Alignment) -> str:
    spot_ids = {s["id"] for s in rois["spots"]}
    labels, warps = [], {}
    for task in tasks:
        blob = task["data"].get("blob")
        found = latest_labels(task, spot_ids) if blob else None
        if found:
            labels.extend([blob, spot, label] for spot, label in found.items())
            m00, m01, m02, m10, m11, m12 = alignment.warp(blob)
            warps[blob] = [round(m00, 4), round(m01, 4), round(m02 * 2) / 2,
                           round(m10, 4), round(m11, 4), round(m12 * 2) / 2]
    payload = {
        "format": FINGERPRINT_FORMAT,
        "labels": sorted(labels),
        "rois": rois,
        "store_margin": STORE_MARGIN,
        "quality_margin": quality_margin,
        "alignment": warps,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def previous_fingerprint(asset_name: str) -> str | None:
    """Fingerprint stored in dataset_info.json of the latest data asset version (None if there is none)."""
    found = latest_asset_file(asset_name, "dataset_info.json")
    if found is None:
        print(f"No previous versions of {asset_name}")
        return None
    content, version = found
    print(f"Latest version: {asset_name}:{version}")
    return json.loads(content).get("fingerprint")


def run_check(rois_path: Path, out: Path, project: int, ls_url: str, token: str,
              asset_name: str, quality_margin: float, force: bool, alignment: Alignment) -> bool:
    rois = json.loads(rois_path.read_text(encoding="utf-8"))
    print(f"Exporting annotations of project {project}...")
    tasks = export_tasks(ls_url.rstrip("/"), token, project)
    out.mkdir(parents=True, exist_ok=True)
    (out / "export.json").write_text(json.dumps(tasks), encoding="utf-8")

    fingerprint = dataset_fingerprint(tasks, rois, quality_margin, alignment)
    previous = previous_fingerprint(asset_name)
    changed = force or fingerprint != previous
    check = {"fingerprint": fingerprint, "previous": previous, "changed": changed, "forced": force,
             "n_tasks": len(tasks), "rois_version": rois["version"]}
    (out / "check.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    print(f"Fingerprint {fingerprint[:12]} (previous {previous[:12] if previous else '-'}) -> changed={changed}")
    return changed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rois", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--alignment", type=Path, default=None, help="align_photos output (alignment.csv)")
    p.add_argument("--project", type=int, required=True)
    p.add_argument("--ls-url", required=True)
    p.add_argument("--keyvault-url", required=True)
    p.add_argument("--secret-name", default="label-studio-token")
    p.add_argument("--asset-name", default="parkwatch-crops")
    p.add_argument("--quality-margin", type=float, default=0.10)
    p.add_argument("--force", default="false", help="true = rebuild even if the fingerprint did not change")
    p.add_argument("--control-output", action="store_true",
                   help="publish `changed` as a pipeline control output (only inside Azure ML)")
    args = p.parse_args()

    token = os.getenv("LS_TOKEN") or get_secret(args.keyvault_url, args.secret_name)
    changed = run_check(args.rois, args.out, args.project, args.ls_url, token,
                        args.asset_name, args.quality_margin, args.force.lower() == "true",
                        Alignment.from_dir(args.alignment))
    if args.control_output:
        write_control_outputs({"changed": changed})


if __name__ == "__main__":
    main()
