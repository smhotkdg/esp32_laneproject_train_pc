from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def read_jsonl(path: str | Path) -> List[Dict]:
    path = Path(path)
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_path(manifest_path: Path, maybe_relative: str | None) -> Optional[Path]:
    if maybe_relative is None or maybe_relative == "":
        return None
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (manifest_path.parent / p).resolve()


def crop_bottom_roi(
    image: np.ndarray,
    lane_mask: np.ndarray,
    drivable_mask: Optional[np.ndarray],
    top_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    h = image.shape[0]
    y0 = max(0, min(h - 1, int(round(h * top_ratio))))
    image = image[y0:, :]
    lane_mask = lane_mask[y0:, :]
    if drivable_mask is not None:
        drivable_mask = drivable_mask[y0:, :]
    return image, lane_mask, drivable_mask


def random_brightness_contrast(image: np.ndarray) -> np.ndarray:
    alpha = random.uniform(0.85, 1.20)
    beta = random.uniform(-18, 18)
    out = image.astype(np.float32) * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def random_gamma(image: np.ndarray) -> np.ndarray:
    gamma = random.uniform(0.85, 1.15)
    table = ((np.arange(256) / 255.0) ** gamma) * 255.0
    table = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


def random_shadow(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x1, x2 = random.randint(0, w - 1), random.randint(0, w - 1)
    poly = np.array([[x1, 0], [x2, h - 1], [w, h - 1], [w, 0]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)

    factor = random.uniform(0.55, 0.85)
    shaded = image.copy().astype(np.float32)
    if shaded.ndim == 3:
        shaded[mask > 0] *= factor
    else:
        shaded[mask > 0] *= factor
    return np.clip(shaded, 0, 255).astype(np.uint8)


def random_blur(image: np.ndarray) -> np.ndarray:
    k = random.choice([0, 0, 3, 5])
    if k <= 1:
        return image
    return cv2.GaussianBlur(image, (k, k), 0)


def random_warp(
    image: np.ndarray,
    lane_mask: np.ndarray,
    drivable_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    h, w = image.shape[:2]
    src = np.float32(
        [
            [0.05 * w, 0.05 * h],
            [0.95 * w, 0.05 * h],
            [0.98 * w, 0.98 * h],
            [0.02 * w, 0.98 * h],
        ]
    )
    jitter = np.float32(
        [
            [random.uniform(-0.03, 0.03) * w, random.uniform(-0.03, 0.03) * h],
            [random.uniform(-0.03, 0.03) * w, random.uniform(-0.03, 0.03) * h],
            [random.uniform(-0.04, 0.04) * w, random.uniform(-0.02, 0.02) * h],
            [random.uniform(-0.04, 0.04) * w, random.uniform(-0.02, 0.02) * h],
        ]
    )
    dst = src + jitter
    M = cv2.getPerspectiveTransform(src, dst)

    image2 = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    lane2 = cv2.warpPerspective(lane_mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    driv2 = None
    if drivable_mask is not None:
        driv2 = cv2.warpPerspective(
            drivable_mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
        )
    return image2, lane2, driv2


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 127).astype(np.float32)


class LaneDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        input_width: int = 160,
        input_height: int = 96,
        input_channels: int = 1,
        train: bool = True,
        roi_top_ratio: float = 0.375,
        roi_jitter: float = 0.04,
        augment: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.rows = read_jsonl(self.manifest_path)
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.train = train
        self.roi_top_ratio = roi_top_ratio
        self.roi_jitter = roi_jitter
        self.augment = augment and train

        if len(self.rows) == 0:
            raise ValueError(f"No samples found in manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _read_image(self, row: Dict) -> np.ndarray:
        image_path = resolve_path(self.manifest_path, row["image"])
        if image_path is None or not image_path.exists():
            raise FileNotFoundError(f"Image not found: {row['image']}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _read_mask(self, row: Dict, key: str) -> Optional[np.ndarray]:
        mask_path = resolve_path(self.manifest_path, row.get(key))
        if mask_path is None:
            return None
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        return mask

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.rows[index]
        image = self._read_image(row)
        lane_mask = self._read_mask(row, "lane_mask")
        if lane_mask is None:
            raise RuntimeError("lane_mask is required for every sample")
        drivable_mask = self._read_mask(row, "drivable_mask")

        top_ratio = self.roi_top_ratio
        if self.train:
            top_ratio += random.uniform(-self.roi_jitter, self.roi_jitter)
            top_ratio = max(0.20, min(0.55, top_ratio))
        image, lane_mask, drivable_mask = crop_bottom_roi(image, lane_mask, drivable_mask, top_ratio)

        if self.augment:
            if random.random() < 0.85:
                image = random_brightness_contrast(image)
            if random.random() < 0.60:
                image = random_gamma(image)
            if random.random() < 0.35:
                image = random_shadow(image)
            if random.random() < 0.25:
                image = random_blur(image)
            if random.random() < 0.45:
                image, lane_mask, drivable_mask = random_warp(image, lane_mask, drivable_mask)

        image = cv2.resize(image, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        lane_mask = cv2.resize(lane_mask, (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)
        if drivable_mask is not None:
            drivable_mask = cv2.resize(
                drivable_mask, (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST
            )

        if self.input_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[..., None]
        elif self.input_channels == 3:
            pass
        else:
            raise ValueError("input_channels must be 1 or 3")

        image = image.astype(np.float32) / 255.0
        lane_mask = normalize_mask(lane_mask)[None, ...]
        if drivable_mask is not None:
            drivable_mask_float = normalize_mask(drivable_mask)[None, ...]
            has_drivable = np.array([1.0], dtype=np.float32)
        else:
            drivable_mask_float = np.zeros_like(lane_mask, dtype=np.float32)
            has_drivable = np.array([0.0], dtype=np.float32)

        image = np.transpose(image, (2, 0, 1))
        sample = {
            "image": torch.from_numpy(image).float(),
            "lane_mask": torch.from_numpy(lane_mask).float(),
            "drivable_mask": torch.from_numpy(drivable_mask_float).float(),
            "has_drivable": torch.from_numpy(has_drivable).float(),
            "source": row.get("source", "unknown"),
        }
        return sample
