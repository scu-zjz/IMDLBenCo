"""APSC-Net adapter for IMDLBenCo.

The architecture follows the authors' CVPR 2024 MIML implementation while
removing its runtime dependency on the legacy MMCV/MMSEG stack.

Reference:
    Qu et al., "Towards Modern Image Manipulation Localization:
    A Large-Scale Dataset and Novel Methods", CVPR 2024.
    https://github.com/qcf-568/MIML

The upstream MIML implementation is distributed under CC BY-NC 4.0. See
``UPSTREAM_NOTICE.md`` in this directory before redistributing this adapter.
"""

import os
import warnings
from collections import OrderedDict
from functools import partial
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath
except ImportError:  # timm < 0.9
    from timm.models.layers import DropPath

from IMDLBenCo.registry import MODELS


def _resize(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ConvModule(nn.Module):
    """Small MMCV ConvModule-compatible subset.

    Attribute names deliberately match MMCV (``conv``, ``bn``, ``activate``)
    so that official state dictionaries can be loaded without rewriting every
    decoder key.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        inplace: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        self.pointwise_conv = ConvModule(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1.0,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class ConvNeXt(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        depths: Sequence[int] = (3, 3, 27, 3),
        dims: Sequence[int] = (128, 256, 512, 1024),
        drop_path_rate: float = 0.4,
        layer_scale_init_value: float = 1.0,
        out_indices: Sequence[int] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("APSC-Net ConvNeXt expects four stages")

        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                )
            ]
        )
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        offset = 0
        self.stages = nn.ModuleList()
        for stage_idx in range(4):
            self.stages.append(
                nn.Sequential(
                    *[
                        ConvNeXtBlock(
                            dims[stage_idx],
                            drop_path=rates[offset + block_idx],
                            layer_scale_init_value=layer_scale_init_value,
                        )
                        for block_idx in range(depths[stage_idx])
                    ]
                )
            )
            offset += depths[stage_idx]

        self.out_indices = tuple(out_indices)
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for stage_idx, dim in enumerate(dims):
            self.add_module(f"norm{stage_idx}", norm_layer(dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outputs = []
        for stage_idx in range(4):
            x = self.downsample_layers[stage_idx](x)
            x = self.stages[stage_idx](x)
            if stage_idx in self.out_indices:
                outputs.append(getattr(self, f"norm{stage_idx}")(x))
        return tuple(outputs)


class PPM(nn.ModuleList):
    def __init__(
        self,
        pool_scales: Sequence[int],
        in_channels: int,
        channels: int,
    ) -> None:
        super().__init__(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    ConvModule(in_channels, channels, 1),
                )
                for scale in pool_scales
            ]
        )

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        return [_resize(module(x), x.shape[-2:]) for module in self]


class DepthwiseSeparableASPPModule(nn.ModuleList):
    def __init__(
        self,
        dilations: Sequence[int],
        in_channels: int,
        channels: int,
    ) -> None:
        modules = []
        for dilation in dilations:
            if dilation == 1:
                modules.append(ConvModule(in_channels, channels, 1))
            else:
                modules.append(
                    DepthwiseSeparableConvModule(
                        in_channels,
                        channels,
                        3,
                        dilation=dilation,
                        padding=dilation,
                    )
                )
        super().__init__(modules)

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        return [module(x) for module in self]


class DepthwiseSeparableASPPHead2(nn.Module):
    def __init__(
        self,
        in_channels: int = 2048,
        channels: int = 512,
        dilations: Sequence[int] = (1, 12, 24, 36),
        c1_in_channels: int = 256,
        c1_channels: int = 48,
        dropout_ratio: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(in_channels, channels, 1),
        )
        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations, in_channels, channels
        )
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * channels, channels, 3, padding=1
        )
        self.c1_bottleneck = ConvModule(c1_in_channels, c1_channels, 1)
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                channels + c1_channels, channels, 3, padding=1
            ),
            DepthwiseSeparableConvModule(channels, channels, 3, padding=1),
        )
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def cls_seg(self, feature: torch.Tensor) -> torch.Tensor:
        return self.conv_seg(self.dropout(feature))

    def forward(
        self, inputs: Sequence[torch.Tensor], trans: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del trans
        low_level, x = inputs
        aspp_outputs = [_resize(self.image_pool(x), x.shape[-2:])]
        aspp_outputs.extend(self.aspp_modules(x))
        output_before_low_level = self.bottleneck(torch.cat(aspp_outputs, dim=1))
        low_level = self.c1_bottleneck(low_level)
        output = _resize(output_before_low_level, low_level.shape[-2:])
        output = self.sep_bottleneck(torch.cat([output, low_level], dim=1))
        return self.cls_seg(output), output_before_low_level


def min_max_norm(x: torch.Tensor) -> torch.Tensor:
    maximum = x.amax(dim=(2, 3), keepdim=True)
    minimum = x.amin(dim=(2, 3), keepdim=True)
    return (x - minimum) / (maximum - minimum + 1e-8)


class HeadAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kg = nn.Sequential(
            nn.UpsamplingBilinear2d(size=(64, 64)),
            ConvModule(1, 32, 5, padding=2, stride=2),
            ConvModule(32, 64, 5, padding=2, stride=2),
            ConvModule(64, 128, 5, padding=2, stride=2),
            ConvModule(128, 256, 5, padding=2, stride=2),
            ConvModule(256, 512, 3, padding=1, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 961),
        )

    def forward(self, attention: torch.Tensor) -> torch.Tensor:
        batch_size = attention.shape[0]
        kernels = self.kg(attention).reshape(batch_size, 1, 31, 31)
        filtered = torch.cat(
            [
                F.conv2d(attention[i:i + 1], kernels[i:i + 1], padding=15)
                for i in range(batch_size)
            ],
            dim=0,
        )
        return torch.maximum(min_max_norm(filtered), attention)


class SCSEModule(nn.Module):
    def __init__(self, in_channels: int = 2560, reduction: int = 8) -> None:
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.c11 = ConvModule(2560, 2048, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c11(x * self.cSE(x))


class APSCDecodeHead(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int] = (128, 256, 512, 1024),
        channels: int = 512,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        if tuple(in_channels) != (128, 256, 512, 1024) or channels != 512:
            raise ValueError(
                "The released APSC decoder is fixed to channels "
                "(128, 256, 512, 1024) and width 512"
            )
        self.in_channels = tuple(in_channels)
        self.channels = channels
        self.psp_modules = PPM(pool_scales, in_channels[-1], channels)
        self.bottleneck = ConvModule(
            in_channels[-1] + len(pool_scales) * channels,
            channels,
            3,
            padding=1,
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        reduction = 8
        self.SE1 = nn.Sequential(
            nn.Conv2d(1024, 1024 // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024 // reduction, 2, 1),
            nn.Sigmoid(),
        )
        self.SE2 = nn.Sequential(
            nn.Conv2d(1536, 1536 // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1536 // reduction, 3, 1),
            nn.Sigmoid(),
        )
        self.SE3 = nn.Sequential(
            nn.Conv2d(2048, 2048 // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048 // reduction, 4, 1),
            nn.Sigmoid(),
        )
        self.lateral_convs = nn.ModuleList(
            [ConvModule(value, channels, 1, inplace=False) for value in in_channels[:-1]]
        )
        self.fpn_convs = nn.ModuleList(
            [ConvModule(channels, channels, 3, padding=1, inplace=False) for _ in range(3)]
        )
        self.MSDEC = DepthwiseSeparableASPPHead2()
        self.convert = nn.Conv2d(512, 256, 1)
        self.ha = HeadAttn()
        self.CE = SCSEModule()
        self.ds = nn.UpsamplingBilinear2d(scale_factor=0.5)
        self.maxp = nn.AdaptiveMaxPool2d(1)
        self.cls_head = nn.Sequential(
            ConvModule(3072, 512, 1, inplace=False),
            nn.MaxPool2d(2, 2),
            ConvModule(512, 256, 3, padding=1, inplace=False),
            nn.MaxPool2d(2, 2),
            ConvModule(256, 256, 3, padding=1, inplace=False),
            nn.AdaptiveMaxPool2d(1),
            nn.Dropout(p=0.2),
            nn.Conv2d(256, 2, 1),
        )
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(channels, 2, kernel_size=1)

    def cls_seg(self, feature: torch.Tensor) -> torch.Tensor:
        return self.conv_seg(self.dropout(feature))

    def psp_forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        x = inputs[-1]
        outputs = [x]
        outputs.extend(self.psp_modules(x))
        return self.bottleneck(torch.cat(outputs, dim=1))

    def forward(
        self, inputs: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, Sequence[torch.Tensor], torch.Tensor]:
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.psp_forward(inputs))

        for i in range(3, 0, -1):
            target_size = laterals[i - 1].shape[-2:]
            if i == 3:
                weights = self.SE1(
                    torch.cat((self.avg(laterals[2]), self.avg(laterals[3])), dim=1)
                )
                laterals[2] = weights[:, 0:1] * laterals[2] + weights[
                    :, 1:2
                ] * _resize(laterals[3], target_size)
            elif i == 2:
                weights = self.SE2(
                    torch.cat(
                        (
                            self.avg(laterals[1]),
                            self.avg(laterals[2]),
                            self.avg(laterals[3]),
                        ),
                        dim=1,
                    )
                )
                laterals[1] = (
                    weights[:, 0:1] * laterals[1]
                    + weights[:, 1:2] * _resize(laterals[2], target_size)
                    + weights[:, 2:3] * _resize(laterals[3], target_size)
                )
            else:
                weights = self.SE3(
                    torch.cat(tuple(self.avg(item) for item in laterals), dim=1)
                )
                laterals[0] = (
                    weights[:, 0:1] * laterals[0]
                    + weights[:, 1:2] * _resize(laterals[1], target_size)
                    + weights[:, 2:3] * _resize(laterals[2], target_size)
                    + weights[:, 3:4] * _resize(laterals[3], target_size)
                )

        fpn_list = [self.fpn_convs[i](laterals[i]) for i in range(3)]
        fpn_list.append(laterals[-1])
        auxiliary_logits = [self.cls_seg(fpn_list[0])]
        low_level = self.convert(fpn_list[0])
        target_size = fpn_list[1].shape[-2:]
        fpn = torch.cat([_resize(item, target_size) for item in fpn_list], dim=1)

        fpn_adds = None
        lab_outs = None
        for iteration in range(3):
            positive_map = F.interpolate(
                F.softmax(auxiliary_logits[-1], dim=1)[:, 1:2],
                scale_factor=0.5,
                mode="bilinear",
                align_corners=False,
            )
            fpn = fpn * self.ha(positive_map)
            lab_outs, fpn_adds = self.MSDEC([low_level, fpn], trans=False)
            if iteration != 2:
                auxiliary_logits.append(lab_outs)
                fpn = self.CE(torch.cat((fpn, fpn_adds), dim=1))

        assert lab_outs is not None and fpn_adds is not None
        batch_size, channels, height, width = fpn_adds.shape
        classifier_input = torch.stack(
            (
                fpn_adds,
                fpn[:, :512],
                fpn[:, 512:1024],
                fpn[:, 1024:1536],
                fpn[:, 1536:2048],
                self.ds(F.softmax(lab_outs, dim=1)[:, 1:2]).expand_as(fpn_adds),
            ),
            dim=2,
        ).reshape(batch_size, channels * 6, height, width)
        classification_logits = self.cls_head(classifier_input.detach())
        return lab_outs, auxiliary_logits, classification_logits


class OhemCrossEntropy(nn.Module):
    """OHEM loss matching the official APSC-Net training configuration."""

    def __init__(
        self,
        ignore_label: int = 100,
        threshold: float = 0.7,
        min_kept: int = 100000,
    ) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.threshold = threshold
        self.min_kept = max(1, min_kept)

    def forward(self, score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = F.softmax(score, dim=1)
        pixel_losses = F.cross_entropy(
            score,
            target,
            ignore_index=self.ignore_label,
            reduction="none",
        ).reshape(-1)
        valid = target.reshape(-1) != self.ignore_label
        safe_target = target.clone()
        safe_target[safe_target == self.ignore_label] = 0
        confidence = probabilities.gather(1, safe_target.unsqueeze(1))
        confidence, indices = confidence.reshape(-1)[valid].sort()
        if confidence.numel() == 0:
            return score.sum() * 0.0
        keep_threshold = max(
            confidence[min(self.min_kept, confidence.numel() - 1)].item(),
            self.threshold,
        )
        selected = pixel_losses[valid][indices][confidence < keep_threshold]
        if selected.numel() == 0:
            selected = pixel_losses[valid][indices][:1]
        return selected.mean()


@MODELS.register_module()
class APSCNet(nn.Module):
    """IMDLBenCo-compatible APSC-Net.

    Inputs are expected to use IMDLBenCo's normal ImageNet normalization.
    ``forward`` returns the standard ``backward_loss``, ``pred_mask`` and
    ``pred_label`` entries used by the training and evaluation runners.
    """

    def __init__(
        self,
        pretrained: str = None,
        backbone_depths=(3, 3, 27, 3),
        drop_path_rate: float = 0.4,
        strict_load: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = ConvNeXt(
            depths=backbone_depths,
            dims=(128, 256, 512, 1024),
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=1.0,
        )
        self.decode_head = APSCDecodeHead()
        self.loss_decode = OhemCrossEntropy()
        self.pretrained = pretrained
        if pretrained:
            self.load_pretrained(pretrained, strict=strict_load)

    @staticmethod
    def _unwrap_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
        if not isinstance(checkpoint, Mapping):
            raise TypeError("Checkpoint must be a mapping")
        for key in ("state_dict", "model", "model_state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                checkpoint = candidate
                break
        if not isinstance(checkpoint, Mapping):
            raise TypeError("No state dictionary found in checkpoint")

        state_dict = OrderedDict()
        for key, value in checkpoint.items():
            if not torch.is_tensor(value):
                continue
            for prefix in ("module.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            state_dict[key] = value
        return state_dict

    def load_pretrained(self, path: str, strict: bool = False):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"APSC-Net checkpoint not found: {path}")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        state_dict = self._unwrap_state_dict(checkpoint)
        incompatible = self.load_state_dict(state_dict, strict=strict)
        if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            warnings.warn(
                "APSC-Net checkpoint loaded non-strictly: "
                f"{len(incompatible.missing_keys)} missing and "
                f"{len(incompatible.unexpected_keys)} unexpected keys.",
                stacklevel=2,
            )
        return incompatible

    def forward(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> Dict[str, object]:
        del label, args, kwargs
        input_size = image.shape[-2:]
        features = self.backbone(image)
        logits, auxiliary_logits, classification_logits = self.decode_head(features)

        resized_logits = _resize(logits, input_size)
        pred_mask = F.softmax(resized_logits, dim=1)[:, 1:2]
        pred_label = F.softmax(classification_logits, dim=1)[:, 1, 0, 0]

        if mask is None:
            total_loss = resized_logits.sum() * 0.0
            final_loss = total_loss
            auxiliary_loss = total_loss
        else:
            target = mask
            if target.ndim == 4:
                target = target[:, 0]
            target = target.long()
            final_loss = self.loss_decode(resized_logits, target)
            auxiliary_loss = sum(
                self.loss_decode(_resize(item, input_size), target)
                for item in auxiliary_logits
            )
            # The released MMSEG head applies a factor of 1/2 to both the
            # final prediction and each auxiliary prediction.
            total_loss = 0.5 * (final_loss + auxiliary_loss)

        return {
            "backward_loss": total_loss,
            "pred_mask": pred_mask,
            "pred_label": pred_label,
            "visual_loss": {
                "loss_ohem": final_loss,
                "loss_aux": auxiliary_loss,
                "combined_loss": total_loss,
            },
            "visual_image": {"pred_mask": pred_mask},
        }
