from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def relpath(path: Path, start: Path) -> str:
    return os.path.relpath(str(path.resolve()), str(start.resolve()))


def stable_split_id(text: str, modulo: int = 100) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % modulo


def load_image_size(path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img.shape[1], img.shape[0]


def lane_thickness_from_width(width: int) -> int:
    # TuSimple images are usually 1280px wide; 5~6 px GT width makes downscaled supervision stabler.
    return max(2, int(round(width / 240.0)))


def resolve_tusimple_image(root: Path, label_parent: Path, raw_file: str) -> Path | None:
    rel = Path(raw_file)
    candidates = [
        root / rel,
        label_parent / rel,
        label_parent.parent / rel,
        root / "train_set" / rel,
        root / "test_set" / rel,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def rasterize_tusimple_lane_mask(item: Dict, image_size: Tuple[int, int]) -> np.ndarray:
    w, h = image_size
    mask = np.zeros((h, w), dtype=np.uint8)
    h_samples = item.get("h_samples", [])
    thickness = lane_thickness_from_width(w)

    for lane in item.get("lanes", []):
        pts: List[List[int]] = []
        for x, y in zip(lane, h_samples):
            if x is None or x < 0:
                continue
            pts.append([int(round(x)), int(round(y))])
        if len(pts) >= 2:
            pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [pts_np], False, 255, thickness=thickness, lineType=cv2.LINE_AA)
    return mask


def prepare_tusimple(root: Path, out_root: Path, val_ratio: float = 0.10) -> Dict[str, List[Dict]]:
    manifest_dir = out_root / "manifests"
    ensure_dir(manifest_dir)

    manifests: Dict[str, List[Dict]] = {"train": [], "val": []}
    label_files = sorted(root.glob("**/label_data*.json"))
    if not label_files:
        raise FileNotFoundError(
            "label_data*.json 파일을 찾지 못했습니다. TuSimple 루트 또는 train_set 폴더를 지정했는지 확인하세요."
        )

    seen_raw_files: set[str] = set()
    skipped_missing_images = 0

    for jf in label_files:
        for line in jf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            raw_file = item.get("raw_file")
            if not raw_file or raw_file in seen_raw_files:
                continue
            seen_raw_files.add(raw_file)

            image_path = resolve_tusimple_image(root, jf.parent, raw_file)
            if image_path is None:
                skipped_missing_images += 1
                continue

            image_size = load_image_size(image_path)
            lane_mask = rasterize_tusimple_lane_mask(item, image_size)

            split = "val" if stable_split_id(raw_file) < int(round(val_ratio * 100.0)) else "train"
            mask_name = f"{Path(raw_file).with_suffix('').as_posix().replace('/', '__')}_lane.png"
            mask_path = out_root / "masks" / "tusimple" / split / mask_name
            ensure_dir(mask_path.parent)
            cv2.imwrite(str(mask_path), lane_mask)

            manifests[split].append(
                {
                    "image": relpath(image_path, manifest_dir),
                    "lane_mask": relpath(mask_path, manifest_dir),
                    "drivable_mask": None,
                    "source": "tusimple",
                    "split": split,
                    "raw_file": raw_file,
                }
            )

    stats = {
        "train": len(manifests["train"]),
        "val": len(manifests["val"]),
        "label_files": [str(x) for x in label_files],
        "skipped_missing_images": skipped_missing_images,
        "val_ratio": val_ratio,
    }
    (out_root / "tusimple_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    write_jsonl(manifest_dir / "tusimple_train.jsonl", manifests["train"])
    write_jsonl(manifest_dir / "tusimple_val.jsonl", manifests["val"])
    return manifests


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare TuSimple into JSONL manifests for ESP32-S3 lane training.")
    p.add_argument("--tusimple-root", required=True, help="TuSimple root or train_set directory")
    p.add_argument("--out", required=True, help="Output folder for manifests and generated masks")
    p.add_argument("--tusimple-val-ratio", type=float, default=0.10, help="Deterministic validation split ratio")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_root = Path(args.out).resolve()
    ensure_dir(out_root)

    manifests = prepare_tusimple(Path(args.tusimple_root).resolve(), out_root, args.tusimple_val_ratio)
    print("Prepared TuSimple only dataset.")
    print(f"Train samples: {len(manifests['train'])}")
    print(f"Val samples:   {len(manifests['val'])}")
    print(f"Saved manifests under: {out_root / 'manifests'}")


if __name__ == "__main__":
    main()
