from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lane_project.models.tiny_lane_net import load_model_from_checkpoint
from lane_project.utils.vision import overlay_lane_result, preprocess_frame




class ONNXLaneRunner:
    def __init__(self, onnx_path: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise SystemExit("onnxruntime가 필요합니다. `pip install onnxruntime` 후 다시 실행하세요.") from exc

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = self.session.run(None, {self.input_name: x[None, ...]})[0]
        return out[0, 0]


class TorchLaneRunner:
    def __init__(self, checkpoint: str, input_channels: int, width_mult: float, aux_drivable: bool) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model_from_checkpoint(
            checkpoint,
            input_channels=input_channels,
            width_mult=width_mult,
            aux_drivable=aux_drivable,
            map_location=self.device,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, x: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(x[None, ...]).float().to(self.device)
        logits = self.model.forward_lane_only(tensor)
        mask = torch.sigmoid(logits)[0, 0].cpu().numpy()
        return mask


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PC webcam lane test (TuSimple-only workflow).")
    p.add_argument("--weights", required=True, help="Path to .pt or .onnx")
    p.add_argument("--source", default="0", help="OpenCV source index or video path")
    p.add_argument("--input-width", type=int, default=160)
    p.add_argument("--input-height", type=int, default=96)
    p.add_argument("--input-channels", type=int, default=1, choices=[1, 3])
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--aux-drivable", action="store_true", help="TuSimple 전용 흐름에서는 보통 사용하지 않습니다.")
    p.add_argument("--roi-top-ratio", type=float, default=0.375)
    p.add_argument("--threshold", type=float, default=0.5)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    weights = str(args.weights)
    if weights.lower().endswith(".onnx"):
        runner = ONNXLaneRunner(weights)
    else:
        runner = TorchLaneRunner(weights, args.input_channels, args.width_mult, args.aux_drivable)

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"비디오 소스를 열 수 없습니다: {args.source}")

    prev_t = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        inp, _, _ = preprocess_frame(
            frame,
            input_width=args.input_width,
            input_height=args.input_height,
            roi_top_ratio=args.roi_top_ratio,
            input_channels=args.input_channels,
        )
        mask = runner(inp)
        vis = overlay_lane_result(frame, mask, roi_top_ratio=args.roi_top_ratio, threshold=args.threshold)

        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

        cv2.putText(vis, f"FPS: {fps:.1f}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("ESP32-S3 Lane Test", vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
