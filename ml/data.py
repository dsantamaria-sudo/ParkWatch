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

from crops import clip_box, expand_box, letterbox

CLASSES = ("free", "occupied")  # order = model class index


def read_manifest(data_dir: Path) -> list[dict]:
    with open(Path(data_dir) / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["bbox"] = tuple(float(r[k]) for k in ("bx0", "by0", "bx1", "by1"))
        r["brightness"], r["contrast"] = float(r["brightness"]), float(r["contrast"])
    return rows


def to_tensor_normalized(image: Image.Image, mean, std) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (x - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)


class SpotCrops(Dataset):
    """Each item: crop with margin (and jitter when augmenting) -> letterbox -> colour/flip augmentation -> tensor."""

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
    ):
        self.root = Path(data_dir)
        self.rows = rows
        self.input_size, self.crop_margin = input_size, crop_margin
        self.mean, self.std = mean, std
        self.augment, self.jitter, self.hflip = augment, jitter, hflip
        self.color = (
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.02) if augment and color_aug else None
        )
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    def __len__(self) -> int:
        return len(self.rows)

    def load_image(self, i: int) -> Image.Image:
        row = self.rows[i]
        image = Image.open(self.root / row["crop"]).convert("RGB")
        box = expand_box(row["bbox"], self.crop_margin, self.jitter if self.augment else 0.0)
        image = letterbox(image.crop(clip_box(box, image.width, image.height)), self.input_size)
        if self.augment:
            if self.color is not None:
                image = self.color(image)
            if self.hflip and random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def __getitem__(self, i: int):
        x = to_tensor_normalized(self.load_image(i), self.mean, self.std)
        return x, self.class_to_idx[self.rows[i]["label"]], i

    def targets(self) -> list[int]:
        return [self.class_to_idx[r["label"]] for r in self.rows]


def channel_stats(data_dir: Path, rows: list[dict], input_size: int, crop_margin: float, max_n: int = 3000, seed: int = 0):
    """Per-channel mean/std of the training crops (no augmentation). No transfer learning, so no ImageNet stats."""
    sample = random.Random(seed).sample(rows, min(max_n, len(rows)))
    ds = SpotCrops(data_dir, sample, input_size, crop_margin)
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
