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

        providers = ["CPUExecutionProvider"]  # GPU 사용 시 ["CUDAExecutionProvider"] 로 변경 가능
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        
        # ONNX 모델의 입력 크기 자동 감지 (형식: [batch, channels, height, width])
        shape = input_info.shape
        if len(shape) == 4:
            self.model_height = int(shape[2])
            self.model_width = int(shape[3])
            print(f"✅ ONNX 모델 입력 크기 감지됨: {self.model_height} x {self.model_width} (Height x Width)")
        else:
            raise ValueError(f"예상치 못한 입력 shape입니다: {shape}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: (C, H, W) 형태
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
    p = argparse.ArgumentParser(description="PC 웹캠 Lane Detection 테스트 (TuSimple-only)")
    p.add_argument("--weights", required=True, help="모델 경로 (.pt 또는 .onnx)")
    p.add_argument("--source", default="0", help="웹캠 번호 또는 영상 파일 경로 (기본: 0)")
    p.add_argument("--input-width", type=int, default=160, help="Torch 모델 사용 시 입력 너비")
    p.add_argument("--input-height", type=int, default=96, help="Torch 모델 사용 시 입력 높이")
    p.add_argument("--input-channels", type=int, default=3, choices=[1, 3], help="입력 채널 수 (기본: 3)")
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--aux-drivable", action="store_true", help="TuSimple에서는 보통 사용 안 함")
    p.add_argument("--roi-top-ratio", type=float, default=0.375, help="ROI 상단 비율")
    p.add_argument("--threshold", type=float, default=0.5, help="레인 마스크 임계값")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    weights = str(args.weights)
    
    # 모델 로드
    if weights.lower().endswith(".onnx"):
        runner = ONNXLaneRunner(weights)
        input_height = runner.model_height
        input_width = runner.model_width
    else:
        runner = TorchLaneRunner(
            weights, 
            args.input_channels, 
            args.width_mult, 
            args.aux_drivable
        )
        input_height = args.input_height
        input_width = args.input_width

    # 웹캠 또는 영상 열기
    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"비디오 소스를 열 수 없습니다: {args.source}")

    print(f"🎥 웹캠 시작 | 입력 크기: {input_height} x {input_width} | 채널: {args.input_channels}")

    prev_t = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("영상 끝 또는 카메라 연결 끊김")
            break

        # 전처리 (자동으로 모델에 맞는 크기로 리사이즈)
        inp, _, _ = preprocess_frame(
            frame,
            input_width=input_width,
            input_height=input_height,
            roi_top_ratio=args.roi_top_ratio,
            input_channels=args.input_channels,
        )

        # 모델 추론
        mask = runner(inp)

        # 결과 시각화
        vis = overlay_lane_result(
            frame, 
            mask, 
            roi_top_ratio=args.roi_top_ratio, 
            threshold=args.threshold
        )

        # FPS 계산
        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

        cv2.putText(vis, f"FPS: {fps:.1f}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("ESP32-S3 Lane Detection Test", vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):   # ESC 또는 q 키로 종료
            break

    cap.release()
    cv2.destroyAllWindows()
    print("프로그램 종료")


if __name__ == "__main__":
    main()