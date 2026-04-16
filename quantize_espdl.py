from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset

from lane_project.data.lane_dataset import LaneDataset, read_jsonl


class CalibrationDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        num_samples: int = 128,
        input_width: int = 160,
        input_height: int = 96,
        input_channels: int = 1,
        roi_top_ratio: float = 0.375,
        seed: int = 42,
    ) -> None:
        rows = read_jsonl(manifest_path)
        indices = list(range(len(rows)))
        random.Random(seed).shuffle(indices)
        indices = indices[: min(num_samples, len(indices))]

        self.base = LaneDataset(
            manifest_path=manifest_path,
            input_width=input_width,
            input_height=input_height,
            input_channels=input_channels,
            train=False,
            roi_top_ratio=roi_top_ratio,
            roi_jitter=0.0,
            augment=False,
        )
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = self.base[self.indices[idx]]
        return item["image"]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quantize ONNX model to ESP-DL .espdl")
    p.add_argument("--onnx", required=True)
    p.add_argument("--calib-manifest", required=True)
    p.add_argument("--output", required=True, help="Output .espdl path")
    p.add_argument("--target", default="esp32s3")
    p.add_argument("--num-bits", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--calib-steps", type=int, default=64)
    p.add_argument("--input-width", type=int, default=160)
    p.add_argument("--input-height", type=int, default=96)
    p.add_argument("--input-channels", type=int, default=1, choices=[1, 3])
    p.add_argument("--roi-top-ratio", type=float, default=0.375)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    try:
        from esp_ppq.api import espdl_quantize_onnx
    except ImportError as exc:
        raise SystemExit(
            "esp-ppq가 설치되어 있지 않습니다. `pip install esp-ppq` 후 다시 실행하세요."
        ) from exc

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    calib_ds = CalibrationDataset(
        manifest_path=args.calib_manifest,
        num_samples=args.num_samples,
        input_width=args.input_width,
        input_height=args.input_height,
        input_channels=args.input_channels,
        roi_top_ratio=args.roi_top_ratio,
        seed=args.seed,
    )

    loader = DataLoader(calib_ds, batch_size=1, shuffle=False)

    def collate_fn(batch):
        x = batch[0]
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return x.to("cpu")

    graph = espdl_quantize_onnx(
        onnx_import_file=args.onnx,
        espdl_export_file=str(output_path),
        calib_dataloader=loader,
        calib_steps=min(args.calib_steps, len(calib_ds)),
        input_shape=[1, args.input_channels, args.input_height, args.input_width],
        target=args.target,
        num_of_bits=args.num_bits,
        collate_fn=collate_fn,
        device="cpu",
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
    )
    print(f"Quantized model exported to: {output_path}")
    print("PC side quantized graph object returned:", type(graph).__name__)


if __name__ == "__main__":
    main()
