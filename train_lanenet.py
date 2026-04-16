from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from lane_project.data.lane_dataset import LaneDataset
from lane_project.models.tiny_lane_net import TinyLaneNet, count_parameters
from lane_project.utils.metrics import AverageMeter, compute_binary_f1_iou


@dataclass
class TrainConfig:
    train_manifest: str
    val_manifest: str
    save_dir: str
    epochs: int = 80
    batch_size: int = 16
    num_workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    input_width: int = 160
    input_height: int = 96
    input_channels: int = 1
    width_mult: float = 1.0
    aux_drivable: bool = False
    roi_top_ratio: float = 0.375
    roi_jitter: float = 0.04
    amp: bool = True
    drivable_loss_weight: float = 0.25
    seed: int = 42
    resume: str = ""


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self._step = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self._step += 1
        # Warm up the decay: start fast (tracks model closely) and gradually
        # increase smoothing toward the configured decay.  Without warmup,
        # decay=0.999 means the shadow retains ~25 % random-init noise after
        # 1400 steps, causing val_f1 to read 0.000 for the first 10-15 epochs.
        decay = min(self.decay, (1 + self._step) / (10 + self._step))
        for key, value in model.state_dict().items():
            if torch.is_floating_point(self.shadow[key]):
                self.shadow[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                self.shadow[key].copy_(value.detach())

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weighted_bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.float()
    probs = torch.sigmoid(logits)

    # emphasize thin lane pixels and boundaries
    edge = F.max_pool2d(target, kernel_size=3, stride=1, padding=1) - F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    edge = (edge > 0.05).float()
    weight = 1.0 + 4.0 * target + 2.0 * edge

    bce = F.binary_cross_entropy_with_logits(logits, target, weight=weight)

    inter = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
    return bce + dice.mean()


def maybe_load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[GradScaler],
    resume_path: str,
    device: torch.device,
) -> int:
    if not resume_path:
        return 0
    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt and scaler.is_enabled():
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0)) + 1


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    cfg: TrainConfig,
    epoch: int,
    best_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "epoch": epoch,
            "best_score": best_score,
            "model_config": {
                "input_channels": getattr(model, "input_channels", cfg.input_channels),
                "width_mult": getattr(model, "width_mult", cfg.width_mult),
                "aux_drivable": getattr(model, "aux_drivable", cfg.aux_drivable),
            },
            "train_config": asdict(cfg),
        },
        str(path),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    cfg: TrainConfig,
    ema: Optional[ModelEMA] = None,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    lane_meter = AverageMeter()
    driv_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"train {epoch:03d}", leave=False)
    for batch in pbar:
        image = batch["image"].to(device, non_blocking=True)
        lane_mask = batch["lane_mask"].to(device, non_blocking=True)
        drivable_mask = batch["drivable_mask"].to(device, non_blocking=True)
        has_drivable = batch["has_drivable"].to(device, non_blocking=True).view(-1, 1, 1, 1)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            outputs = model(image)
            lane_loss = weighted_bce_dice_loss(outputs["lane"], lane_mask)
            loss = lane_loss
            driv_loss_value = torch.tensor(0.0, device=device)
            if "drivable" in outputs:
                valid_drivable = has_drivable.expand_as(drivable_mask)
                if valid_drivable.sum() > 0:
                    driv_logits = outputs["drivable"]
                    masked_logits = driv_logits[valid_drivable > 0].view(-1, 1)
                    masked_target = drivable_mask[valid_drivable > 0].view(-1, 1)
                    driv_loss_value = F.binary_cross_entropy_with_logits(masked_logits, masked_target)
                    loss = loss + cfg.drivable_loss_weight * driv_loss_value

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        bsz = image.shape[0]
        loss_meter.update(loss.item(), bsz)
        lane_meter.update(lane_loss.item(), bsz)
        driv_meter.update(driv_loss_value.item(), bsz)
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", lane=f"{lane_meter.avg:.4f}")

    return {
        "loss": loss_meter.avg,
        "lane_loss": lane_meter.avg,
        "drivable_loss": driv_meter.avg,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    desc: str = "val",
) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    f1_meter = AverageMeter()
    iou_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"{desc} {epoch:03d}", leave=False)
    for batch in pbar:
        image = batch["image"].to(device, non_blocking=True)
        lane_mask = batch["lane_mask"].to(device, non_blocking=True)

        outputs = model(image)
        lane_logits = outputs["lane"]
        loss = weighted_bce_dice_loss(lane_logits, lane_mask)

        metric = compute_binary_f1_iou(lane_logits, lane_mask, threshold=0.5)
        bsz = image.shape[0]
        loss_meter.update(loss.item(), bsz)
        f1_meter.update(metric["f1"], bsz)
        iou_meter.update(metric["iou"], bsz)
        pbar.set_postfix(f1=f"{f1_meter.avg:.4f}", iou=f"{iou_meter.avg:.4f}")

    return {
        "loss": loss_meter.avg,
        "f1": f1_meter.avg,
        "iou": iou_meter.avg,
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train TinyLaneNet for ESP32-S3 lane segmentation (TuSimple-only workflow).")
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--input-width", type=int, default=160)
    p.add_argument("--input-height", type=int, default=96)
    p.add_argument("--input-channels", type=int, default=1, choices=[1, 3])
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--aux-drivable", action="store_true", help="TuSimple 전용 흐름에서는 보통 사용하지 않습니다.")
    p.add_argument("--roi-top-ratio", type=float, default=0.375)
    p.add_argument("--roi-jitter", type=float, default=0.04)
    p.add_argument("--drivable-loss-weight", type=float, default=0.25)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default="")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = TrainConfig(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        input_width=args.input_width,
        input_height=args.input_height,
        input_channels=args.input_channels,
        width_mult=args.width_mult,
        aux_drivable=args.aux_drivable,
        roi_top_ratio=args.roi_top_ratio,
        roi_jitter=args.roi_jitter,
        amp=not args.no_amp,
        drivable_loss_weight=args.drivable_loss_weight,
        seed=args.seed,
        resume=args.resume,
    )

    set_seed(cfg.seed)
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        cpu_threads = max(1, min(8, os.cpu_count() or 1))
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    print("Workflow: TuSimple-only")

    train_ds = LaneDataset(
        cfg.train_manifest,
        input_width=cfg.input_width,
        input_height=cfg.input_height,
        input_channels=cfg.input_channels,
        train=True,
        roi_top_ratio=cfg.roi_top_ratio,
        roi_jitter=cfg.roi_jitter,
        augment=True,
    )
    val_ds = LaneDataset(
        cfg.val_manifest,
        input_width=cfg.input_width,
        input_height=cfg.input_height,
        input_channels=cfg.input_channels,
        train=False,
        roi_top_ratio=cfg.roi_top_ratio,
        roi_jitter=0.0,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, cfg.batch_size // 2),
        shuffle=False,
        num_workers=max(1, cfg.num_workers // 2),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    resume_model_cfg = {}
    if cfg.resume:
        resume_ckpt = torch.load(cfg.resume, map_location="cpu")
        resume_model_cfg = resume_ckpt.get("model_config", {})

    model = TinyLaneNet(
        input_channels=resume_model_cfg.get("input_channels", cfg.input_channels),
        width_mult=resume_model_cfg.get("width_mult", cfg.width_mult),
        aux_drivable=resume_model_cfg.get("aux_drivable", cfg.aux_drivable),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.05)
    scaler = GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))

    start_epoch = maybe_load_checkpoint(model, optimizer, scaler, cfg.resume, device)
    ema = ModelEMA(model, decay=0.999)

    meta = {
        "params": count_parameters(model),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "device": str(device),
        "config": asdict(cfg),
        "effective_model_config": {
            "input_channels": model.input_channels,
            "width_mult": model.width_mult,
            "aux_drivable": model.aux_drivable,
        },
    }
    (save_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    best_score = -1.0
    history = []

    print(f"Device: {device}")
    if device.type == "cpu":
        print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"Trainable params: {meta['params']:,}")

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, cfg, ema)
        scheduler.step()

        eval_model = TinyLaneNet(
            input_channels=model.input_channels,
            width_mult=model.width_mult,
            aux_drivable=model.aux_drivable,
        ).to(device)
        ema.copy_to(eval_model)

        val_stats = validate(eval_model, val_loader, device, epoch, desc="val")
        score = 0.7 * val_stats["f1"] + 0.3 * val_stats["iou"]
        best_score = max(best_score, score)

        row = {
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats,
            "score": score,
            "best_score": best_score,
            "time_sec": time.time() - t0,
        }
        history.append(row)
        (save_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_stats['loss']:.4f} "
            f"val_f1={val_stats['f1']:.4f} "
            f"val_iou={val_stats['iou']:.4f} "
            f"score={score:.4f} "
            f"best={best_score:.4f}"
        )

        # save current raw model
        save_checkpoint(save_dir / "last.pt", model, optimizer, scheduler, scaler, cfg, epoch, best_score)

        if score >= best_score - 1e-12:
            save_checkpoint(save_dir / "best.pt", eval_model, optimizer, scheduler, scaler, cfg, epoch, best_score)

    print(f"Training finished. Best score: {best_score:.4f}")
    print(f"Artifacts saved to: {save_dir}")


if __name__ == "__main__":
    main()
