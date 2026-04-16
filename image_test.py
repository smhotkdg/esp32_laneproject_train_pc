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
    p = argparse.ArgumentParser(description="Image-based lane detection and saving.")
    p.add_argument("--weights", required=True, help="Path to .pt or .onnx")
    p.add_argument("--source", required=True, help="이미지 파일 경로 또는 폴더 경로")
    p.add_argument("--output", default="runs/detect", help="결과 이미지를 저장할 폴더")
    p.add_argument("--input-width", type=int, default=160)
    p.add_argument("--input-height", type=int, default=96)
    p.add_argument("--input-channels", type=int, default=1, choices=[1, 3])
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--aux-drivable", action="store_true")
    p.add_argument("--roi-top-ratio", type=float, default=0.375)
    p.add_argument("--threshold", type=float, default=0.5)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    # 1. 모델 준비
    weights = str(args.weights)
    if weights.lower().endswith(".onnx"):
        runner = ONNXLaneRunner(weights)
    else:
        runner = TorchLaneRunner(weights, args.input_channels, args.width_mult, args.aux_drivable)

    # 2. 입출력 경로 설정
    source_path = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 폴더인 경우 내부의 이미지들을, 파일인 경우 단일 파일을 리스트에 담음
    if source_path.is_dir():
        image_files = list(source_path.glob("*.[jJ][pP][gG]")) + list(source_path.glob("*.[pP][nN][gG]"))
    else:
        image_files = [source_path]

    if not image_files:
        print(f"처리할 이미지를 찾을 수 없습니다: {args.source}")
        return

    print(f"총 {len(image_files)}개의 이미지를 처리합니다.")

    # 3. 이미지 처리 루프
    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"이미지를 읽을 수 없습니다: {img_path}")
            continue

        # 전처리 및 추론
        inp, _, _ = preprocess_frame(
            frame,
            input_width=args.input_width,
            input_height=args.input_height,
            roi_top_ratio=args.roi_top_ratio,
            input_channels=args.input_channels,
        )
        
        start_t = time.time()
        mask = runner(inp)
        dt = time.time() - start_t

        # 결과 시각화
        vis = overlay_lane_result(frame, mask, roi_top_ratio=args.roi_top_ratio, threshold=args.threshold)

        # 저장 파일명 결정 (원본이름_result.jpg)
        save_path = output_dir / f"{img_path.stem}_result.jpg"
        cv2.imwrite(str(save_path), vis)
        
        print(f"Saved: {save_path.name} ({dt*1000:.1f}ms)")

    print(f"\n모든 작업이 완료되었습니다. 결과 저장 위치: {output_dir}")


if __name__ == "__main__":
    main()