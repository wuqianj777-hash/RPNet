from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .ccpd import CLASS_COUNTS


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(1)
    return torch.stack((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2), dim=1).clamp(0, 1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(1)
    return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=1)


class RPNet8(nn.Module):
    """A compact RPNet-inspired detector with multi-level ROI recognition."""

    def __init__(self, roi_size: tuple[int, int] = (8, 20)) -> None:
        super().__init__()
        self.roi_size = roi_size
        self.stem = ConvBNAct(3, 32, 2)
        self.stage1 = DepthwiseBlock(32, 64, 2)
        self.stage2 = nn.Sequential(DepthwiseBlock(64, 96, 2), DepthwiseBlock(96, 96))
        self.stage3 = nn.Sequential(DepthwiseBlock(96, 160, 2), DepthwiseBlock(160, 160))
        self.stage4 = nn.Sequential(DepthwiseBlock(160, 256, 2), DepthwiseBlock(256, 256))
        self.detector = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 7)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 7, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(64 + 96 + 160, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            DepthwiseBlock(256, 256),
            nn.AdaptiveAvgPool2d((1, 8)),
        )
        self.character_heads = nn.ModuleList(nn.Linear(256 * 8, count) for count in CLASS_COUNTS)

    def extract_features(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        value = self.stem(image)
        c1 = self.stage1(value)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return c1, c2, c3, c4

    def predict_boxes(self, deepest: torch.Tensor) -> torch.Tensor:
        raw = torch.sigmoid(self.detector(deepest))
        # Plate boxes are small; retaining a nonzero minimum avoids collapsed ROIs early in training.
        center = raw[:, :2]
        size = 0.03 + 0.67 * raw[:, 2:]
        return cxcywh_to_xyxy(torch.cat((center, size), dim=1))

    def _roi_pool(self, feature: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        batch = boxes.shape[0]
        out_h, out_w = self.roi_size
        ys = torch.linspace(0, 1, out_h, device=feature.device, dtype=feature.dtype).view(1, out_h, 1)
        xs = torch.linspace(0, 1, out_w, device=feature.device, dtype=feature.dtype).view(1, 1, out_w)
        x1, y1, x2, y2 = boxes.unbind(1)
        grid_x = x1.view(batch, 1, 1) + xs * (x2 - x1).view(batch, 1, 1)
        grid_y = y1.view(batch, 1, 1) + ys * (y2 - y1).view(batch, 1, 1)
        grid = torch.stack((grid_x.expand(-1, out_h, -1), grid_y.expand(-1, -1, out_w)), dim=-1)
        return F.grid_sample(feature, grid.mul(2).sub(1), mode="bilinear", align_corners=True)

    def recognize(self, features: tuple[torch.Tensor, ...], boxes: torch.Tensor) -> list[torch.Tensor]:
        pooled = [self._roi_pool(feature, boxes) for feature in features[:3]]
        value = self.fusion(torch.cat(pooled, dim=1)).flatten(1)
        return [head(value) for head in self.character_heads]

    def forward(
        self, image: torch.Tensor, roi_boxes: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor, ...]]:
        features = self.extract_features(image)
        boxes = self.predict_boxes(features[-1])
        # The recognition branch learns from detector ROIs, while coordinate gradients are
        # stopped so classification cannot move the box away from its localization target.
        recognition_boxes = boxes.detach() if roi_boxes is None else roi_boxes
        logits = self.recognize(features, recognition_boxes)
        return boxes, logits, features
