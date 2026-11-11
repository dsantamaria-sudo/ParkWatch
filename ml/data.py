"""Crop dataset built from manifest.csv (build_dataset.py) and shared preprocessing."""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from crops import clip_box, expand_box, grey_outside, letterbox, polygon_mask

CLASSES = ("free", "occupied")  # order = model class index


def read_manifest(data_dir: Path) -> list[dict]:
    with open(Path(data_dir) / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["bbox"] = tuple(float(r[k]) for k in ("bx0", "by0", "bx1", "by1"))
        r["brightness"], r["contrast"] = float(r["brightness"]), float(r["contrast"])
    return rows


def to_tensor_normalized(image: Image.Image, mean, std, mask: Image.Image | None = None) -> torch.Tensor:
    """RGB normalised with mean/std; with `mask` (mode "L") a 4th channel: +1 inside the spot, -1 outside."""
    x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    x = (x - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
    if mask is not None:
        m = torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0)[None]
        x = torch.cat([x, m * 2 - 1])
    return x


class SpotCrops(Dataset):
    """Each item: crop with margin (and jitter when augmenting) -> letterbox -> colour/flip augmentation -> tensor.

    polygons ({spot: polygon in reference px}, crops.spot_polygons) build the spot mask, cropped,
    letterboxed and flipped exactly like the image. It is used by
      mask_channel: the mask as a 4th channel (experiment B)
      outside_grey: everything outside the spot painted grey before augmenting (experiment C)"""

    def __init__(
        self,
        data_dir: Path,
        rows: list[dict],
        input_size: int,
        crop_margin: float,
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
        augment: bool = False,
        jitter: float = 0.0,
        color_aug: bool = True,
        hflip: bool = True,
        polygons: dict[str, list[list[float]]] | None = None,
        mask_channel: bool = False,
        outside_grey: bool = False,
    ):
        self.root = Path(data_dir)
        self.rows = rows
        self.input_size, self.crop_margin = input_size, crop_margin
        self.mean, self.std = mean, std
        self.augment, self.jitter, self.hflip = augment, jitter, hflip
        self.polygons, self.mask_channel, self.outside_grey = polygons, mask_channel, outside_grey
        if (mask_channel or outside_grey) and not polygons:
            raise ValueError("mask_channel / outside_grey need the spot polygons")
        self.color = (
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.02) if augment and color_aug else None
        )
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    def __len__(self) -> int:
        return len(self.rows)

    def mask_of(self, row: dict, size: tuple[int, int]) -> Image.Image:
        """Spot mask on the stored crop: the polygon shifted by the crop origin (bbox minus bx0/by0)."""
        poly = self.polygons[row["spot"]]
        x0, y0 = min(x for x, _ in poly), min(y for _, y in poly)  # bbox corner in reference px
        return polygon_mask(size, poly, (x0 - row["bbox"][0], y0 - row["bbox"][1]))

    def load(self, i: int) -> tuple[Image.Image, Image.Image | None]:
        """(image, mask or None), with the same crop, letterbox and flip."""
        row = self.rows[i]
        image = Image.open(self.root / row["crop"]).convert("RGB")
        mask = self.mask_of(row, image.size) if self.mask_channel or self.outside_grey else None
        box = clip_box(expand_box(row["bbox"], self.crop_margin, self.jitter if self.augment else 0.0), image.width, image.height)
        image = letterbox(image.crop(box), self.input_size)
        mask = letterbox(mask.crop(box), self.input_size) if mask is not None else None
        if self.outside_grey:
            image = grey_outside(image, mask)
        if self.augment:
            if self.color is not None:
                image = self.color(image)
            if self.hflip and random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT) if mask is not None else None
        return image, mask

    def load_image(self, i: int) -> Image.Image:
        return self.load(i)[0]

    def __getitem__(self, i: int):
        image, mask = self.load(i)
        x = to_tensor_normalized(image, self.mean, self.std, mask if self.mask_channel else None)
        return x, self.class_to_idx[self.rows[i]["label"]], i

    def targets(self) -> list[int]:
        return [self.class_to_idx[r["label"]] for r in self.rows]


def channel_stats(data_dir: Path, rows: list[dict], input_size: int, crop_margin: float, max_n: int = 3000, seed: int = 0,
                  **spot_options):
    """Per-channel mean/std of the training crops (no augmentation). No transfer learning, so no ImageNet stats.
    spot_options: polygons / outside_grey of SpotCrops (the grey outside the spot counts in the stats)."""
    sample = random.Random(seed).sample(rows, min(max_n, len(rows)))
    ds = SpotCrops(data_dir, sample, input_size, crop_margin,
                   polygons=spot_options.get("polygons"), outside_grey=spot_options.get("outside_grey", False))
    acc = np.zeros(3)
    acc_sq = np.zeros(3)
    n = 0
    for i in range(len(ds)):
        x = np.asarray(ds.load_image(i), dtype=np.float64) / 255.0
        acc += x.mean((0, 1))
        acc_sq += (x**2).mean((0, 1))
        n += 1
    mean = acc / n
    std = np.sqrt(np.maximum(acc_sq / n - mean**2, 1e-8))
    return tuple(mean.round(4).tolist()), tuple(std.round(4).tolist())
