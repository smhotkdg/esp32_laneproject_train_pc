from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.use_act = act
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def fused(self) -> nn.Sequential:
        fused_conv = fuse_conv_bn_eval(self.conv, self.bn)
        act = nn.ReLU(inplace=False) if self.use_act else nn.Identity()
        return nn.Sequential(fused_conv, act)


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = ConvBNAct(in_ch, in_ch, k=3, s=stride, groups=in_ch)
        self.pw = ConvBNAct(in_ch, out_ch, k=1, s=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class ResidualDS(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = DepthwiseSeparable(channels, channels, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DecoderFuse(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = ConvBNAct(in_ch, out_ch, k=1, s=1, p=0)
        self.fuse = nn.Sequential(
            DepthwiseSeparable(out_ch + skip_ch, out_ch, stride=1),
            ResidualDS(out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj(x)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class SegHead(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparable(in_ch, in_ch, stride=1),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        x = self.block(x)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x


class TinyLaneNet(nn.Module):
    """
    Tiny segmentation-style lane detector designed to survive ESP32-S3 deployment:
    - only standard Conv/DepthwiseConv/Add/Concat/Resize/ReLU style operators
    - optional auxiliary drivable-area head for stronger context during training
    """

    def __init__(
        self,
        input_channels: int = 1,
        width_mult: float = 1.0,
        aux_drivable: bool = False,
    ) -> None:
        super().__init__()
        c1 = _make_divisible(16 * width_mult)
        c2 = _make_divisible(24 * width_mult)
        c3 = _make_divisible(32 * width_mult)
        c4 = _make_divisible(48 * width_mult)
        c5 = _make_divisible(64 * width_mult)

        self.input_channels = input_channels
        self.width_mult = width_mult
        self.aux_drivable = aux_drivable

        self.stem = nn.Sequential(
            ConvBNAct(input_channels, c1, k=3, s=2),
            ResidualDS(c1),
        )
        self.enc1 = nn.Sequential(
            DepthwiseSeparable(c1, c2, stride=2),
            ResidualDS(c2),
        )
        self.enc2 = nn.Sequential(
            DepthwiseSeparable(c2, c3, stride=2),
            ResidualDS(c3),
        )
        self.enc3 = nn.Sequential(
            DepthwiseSeparable(c3, c4, stride=2),
            ResidualDS(c4),
        )
        self.bottleneck = nn.Sequential(
            DepthwiseSeparable(c4, c5, stride=1),
            ResidualDS(c5),
            ResidualDS(c5),
        )

        self.dec3 = DecoderFuse(c5, c3, c3)
        self.dec2 = DecoderFuse(c3, c2, c2)
        self.dec1 = DecoderFuse(c2, c1, c1)

        self.lane_head = SegHead(c1, 1)
        self.drivable_head = SegHead(c1, 1) if aux_drivable else None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out_size = x.shape[-2:]

        s1 = self.stem(x)     # 1/2
        s2 = self.enc1(s1)    # 1/4
        s3 = self.enc2(s2)    # 1/8
        s4 = self.enc3(s3)    # 1/16
        b = self.bottleneck(s4)

        d3 = self.dec3(b, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        outputs: Dict[str, torch.Tensor] = {"lane": self.lane_head(d1, out_size)}
        if self.drivable_head is not None:
            outputs["drivable"] = self.drivable_head(d1, out_size)
        return outputs

    def forward_lane_only(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["lane"]


class LaneOnlyWrapper(nn.Module):
    def __init__(self, model: TinyLaneNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_lane_only(x)


def _make_divisible(v: float, divisor: int = 8, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


def fuse_conv_bn_eval(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    if conv.training or bn.training:
        raise ValueError("Fusion only supports eval() modules.")

    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    )

    conv_weight = conv.weight.detach().clone().view(conv.out_channels, -1)
    if conv.bias is None:
        conv_bias = torch.zeros(conv.out_channels, device=conv_weight.device, dtype=conv_weight.dtype)
    else:
        conv_bias = conv.bias.detach().clone()

    bn_weight = bn.weight.detach().clone()
    bn_bias = bn.bias.detach().clone()
    running_mean = bn.running_mean.detach().clone()
    running_var = bn.running_var.detach().clone()
    eps = bn.eps

    std = torch.sqrt(running_var + eps)
    scale = (bn_weight / std).reshape(-1, 1)

    fused_weight = (conv_weight * scale).view(fused.weight.shape)
    fused_bias = bn_bias + (conv_bias - running_mean) * (bn_weight / std)

    fused.weight.data.copy_(fused_weight)
    fused.bias.data.copy_(fused_bias)
    return fused


def fuse_model_for_export(model: nn.Module) -> nn.Module:
    model = copy.deepcopy(model).eval()

    def _recursive_fuse(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, ConvBNAct):
                setattr(module, name, child.fused())
            else:
                _recursive_fuse(child)

    _recursive_fuse(model)
    return model


def load_model_from_checkpoint(
    checkpoint_path: str,
    input_channels: int = 1,
    width_mult: float = 1.0,
    aux_drivable: bool = False,
    map_location: str | torch.device = "cpu",
) -> TinyLaneNet:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    cfg = checkpoint.get("model_config", {})
    model = TinyLaneNet(
        input_channels=cfg.get("input_channels", input_channels),
        width_mult=cfg.get("width_mult", width_mult),
        aux_drivable=cfg.get("aux_drivable", aux_drivable),
    )
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
