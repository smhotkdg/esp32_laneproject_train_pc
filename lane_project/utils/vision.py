from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def crop_bottom_roi_frame(frame: np.ndarray, roi_top_ratio: float = 0.375) -> Tuple[np.ndarray, int]:
    h = frame.shape[0]
    y0 = max(0, min(h - 1, int(round(h * roi_top_ratio))))
    return frame[y0:, :], y0


def preprocess_frame(
    frame_bgr: np.ndarray,
    input_width: int = 160,
    input_height: int = 96,
    roi_top_ratio: float = 0.375,
    input_channels: int = 1,
) -> Tuple[np.ndarray, np.ndarray, int]:
    roi_bgr, y0 = crop_bottom_roi_frame(frame_bgr, roi_top_ratio=roi_top_ratio)
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(roi_rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    if input_channels == 1:
        inp = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)[None, ...]
    else:
        inp = np.transpose(resized, (2, 0, 1))
    inp = inp.astype(np.float32) / 255.0
    return inp, roi_bgr, y0


def mask_to_left_right_points(mask: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    h, w = mask.shape
    center_x = w // 2
    left_points: List[Tuple[int, int]] = []
    right_points: List[Tuple[int, int]] = []

    for y in range(h - 1, int(h * 0.20), -3):
        xs = np.where(mask[y] > 0)[0]
        if xs.size == 0:
            continue

        left_candidates = xs[xs < center_x]
        right_candidates = xs[xs >= center_x]

        if left_candidates.size > 0:
            left_points.append((int(left_candidates.max()), y))
        if right_candidates.size > 0:
            right_points.append((int(right_candidates.min()), y))
    return left_points, right_points


def fit_polyline(points: List[Tuple[int, int]], height: int) -> Optional[np.ndarray]:
    if len(points) < 6:
        return None
    pts = np.array(points, dtype=np.float32)
    ys = pts[:, 1]
    xs = pts[:, 0]
    try:
        coeff = np.polyfit(ys, xs, deg=2)
    except np.linalg.LinAlgError:
        return None
    poly_points = []
    for y in range(height - 1, int(height * 0.20), -4):
        x = coeff[0] * y * y + coeff[1] * y + coeff[2]
        poly_points.append([int(round(x)), int(y)])
    return np.array(poly_points, dtype=np.int32)


def overlay_lane_result(
    frame_bgr: np.ndarray,
    lane_mask_small: np.ndarray,
    roi_top_ratio: float = 0.375,
    threshold: float = 0.5,
) -> np.ndarray:
    overlay = frame_bgr.copy()
    roi_bgr, y0 = crop_bottom_roi_frame(overlay, roi_top_ratio=roi_top_ratio)
    mask = (lane_mask_small >= threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask, (roi_bgr.shape[1], roi_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    colored = np.zeros_like(roi_bgr)
    colored[..., 1] = mask
    roi_bgr = cv2.addWeighted(roi_bgr, 1.0, colored, 0.30, 0.0)

    left_points, right_points = mask_to_left_right_points(mask)
    left_poly = fit_polyline(left_points, mask.shape[0])
    right_poly = fit_polyline(right_points, mask.shape[0])

    if left_poly is not None:
        cv2.polylines(roi_bgr, [left_poly], isClosed=False, color=(0, 0, 255), thickness=2)
    if right_poly is not None:
        cv2.polylines(roi_bgr, [right_poly], isClosed=False, color=(255, 0, 0), thickness=2)

    if left_poly is not None and right_poly is not None:
        bottom_left = left_poly[0]
        bottom_right = right_poly[0]
        lane_center = int((bottom_left[0] + bottom_right[0]) / 2)
        cv2.line(roi_bgr, (lane_center, roi_bgr.shape[0] - 1), (lane_center, roi_bgr.shape[0] - 25), (0, 255, 255), 2)
        frame_center = roi_bgr.shape[1] // 2
        cv2.line(roi_bgr, (frame_center, roi_bgr.shape[0] - 1), (frame_center, roi_bgr.shape[0] - 25), (255, 255, 0), 2)
        offset = lane_center - frame_center
        cv2.putText(
            roi_bgr,
            f"offset(px): {offset:+d}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    overlay[y0:, :] = roi_bgr
    return overlay
