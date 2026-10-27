from __future__ import annotations

import torch


def box_iou_aligned(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(prediction[:, :2], target[:, :2])
    bottom_right = torch.minimum(prediction[:, 2:], target[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(1)
    pred_area = (prediction[:, 2:] - prediction[:, :2]).clamp(min=0).prod(1)
    target_area = (target[:, 2:] - target[:, :2]).clamp(min=0).prod(1)
    return intersection / (pred_area + target_area - intersection).clamp(min=1e-7)


def logits_to_plate(logits: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([value.argmax(1) for value in logits], dim=1)

