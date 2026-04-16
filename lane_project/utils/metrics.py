from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AverageMeter:
    value: float = 0.0
    avg: float = 0.0
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.value = float(value)
        self.total += float(value) * n
        self.count += int(n)
        self.avg = self.total / max(1, self.count)


def compute_binary_f1_iou(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    target = (target >= 0.5).float()

    tp = (pred * target).sum().item()
    fp = (pred * (1.0 - target)).sum().item()
    fn = ((1.0 - pred) * target).sum().item()
    union = ((pred + target) > 0).float().sum().item()
    inter = (pred * target).sum().item()

    precision = tp / max(1e-6, tp + fp)
    recall = tp / max(1e-6, tp + fn)
    f1 = (2.0 * precision * recall) / max(1e-6, precision + recall)
    iou = inter / max(1e-6, union)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }
