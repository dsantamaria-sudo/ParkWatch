"""ParkNet inference on a full photo: same crop and preprocessing as in training.

Usage:
  python inference.py --model-dir outputs/model --rois ../label_studio/rois.json --image photo.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from alignment import IDENTITY, Warp, align_image
from crops import crop_spot, letterbox
from data import to_tensor_normalized
from parknet import ParkNet


class ParkNetPredictor:
    def __init__(self, model_dir: Path, device: str = "cpu"):
        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
        c = self.config
        self.model = ParkNet(len(c["classes"]), c["width"], c["dropout"])
        self.model.load_state_dict(torch.load(model_dir / "parknet.pt", map_location=device))
        self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def predict(self, image: Image.Image, rois: dict, warp: Warp = IDENTITY) -> dict[str, dict[str, float]]:
        """{spot: {class: probability}} for every spot in rois.json.

        `warp`: drift correction of the photo (alignment.py), the same one used in build_dataset."""
        c = self.config
        if rois["version"] != c["rois_version"]:
            print(f"warning: ROIs {rois['version']} != training ROIs {c['rois_version']}")
        image = align_image(image.convert("RGB"), warp)
        batch = torch.stack(
            [
                to_tensor_normalized(letterbox(crop_spot(image, s["points"], c["crop_margin"]), c["input_size"]), c["norm_mean"], c["norm_std"])
                for s in rois["spots"]
            ]
        ).to(self.device)
        probs = torch.softmax(self.model(batch), dim=1).cpu().tolist()
        return {s["id"]: dict(zip(c["classes"], p)) for s, p in zip(rois["spots"], probs)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--rois", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    args = p.parse_args()
    predictor = ParkNetPredictor(args.model_dir)
    out = predictor.predict(Image.open(args.image), json.loads(args.rois.read_text(encoding="utf-8")))
    for spot, probs in out.items():
        best = max(probs, key=probs.get)
        print(f"{spot}: {best} ({probs[best]:.3f})")


if __name__ == "__main__":
    main()
