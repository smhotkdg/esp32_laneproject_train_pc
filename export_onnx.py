from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lane_project.models.tiny_lane_net import LaneOnlyWrapper, fuse_model_for_export, load_model_from_checkpoint


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export TinyLaneNet to ONNX for ESP-DL (TuSimple-only workflow).")
    p.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt")
    p.add_argument("--output", required=True, help="Output .onnx path")
    p.add_argument("--input-width", type=int, default=160)
    p.add_argument("--input-height", type=int, default=96)
    p.add_argument("--input-channels", type=int, default=1, choices=[1, 3])
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--aux-drivable", action="store_true", help="TuSimple 전용 흐름에서는 보통 사용하지 않습니다.")
    p.add_argument("--opset", type=int, default=18)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)


    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise SystemExit("onnx 패키지가 필요합니다. `pip install onnx` 후 다시 실행하세요.") from exc

    model = load_model_from_checkpoint(
        args.checkpoint,
        input_channels=args.input_channels,
        width_mult=args.width_mult,
        aux_drivable=args.aux_drivable,
        map_location="cpu",
    )
    fused = fuse_model_for_export(model)
    export_model = LaneOnlyWrapper(fused).eval()

    dummy = torch.randn(1, args.input_channels, args.input_height, args.input_width, dtype=torch.float32)

    torch.onnx.export(
        export_model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["lane"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )

    print(f"Exported ONNX to: {output_path}")


if __name__ == "__main__":
    main()
