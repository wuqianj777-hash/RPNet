from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import box_iou_aligned, logits_to_plate
from .model import RPNet8, xyxy_to_cxcywh


def character_loss(logits: list[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    return sum(F.cross_entropy(value, target[:, index]) for index, value in enumerate(logits)) / len(logits)


def train_epoch(
    model: RPNet8,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    box_loss_weight: float,
    detection_only: bool = False,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "box_loss": 0.0, "char_loss": 0.0}
    samples = 0
    for batch in tqdm(loader, desc="train", leave=False, disable=True):
        image = batch["image"].to(device, non_blocking=True)
        target_box = batch["box"].to(device, non_blocking=True)
        target_plate = batch["plate"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            # After localization pretraining, recognition must see the detector's own ROI.
            # This closes the train/deployment gap and lets classification gradients refine the box.
            pred_box, logits, _ = model(image, roi_boxes=target_box if detection_only else None)
            box_loss = F.smooth_l1_loss(xyxy_to_cxcywh(pred_box), xyxy_to_cxcywh(target_box))
            cls_loss = torch.zeros((), device=device) if detection_only else character_loss(logits, target_plate)
            loss = box_loss_weight * box_loss + cls_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        count = image.shape[0]
        samples += count
        totals["loss"] += float(loss.detach()) * count
        totals["box_loss"] += float(box_loss.detach()) * count
        totals["char_loss"] += float(cls_loss.detach()) * count
    return {key: value / samples for key, value in totals.items()}


@torch.inference_mode()
def evaluate(
    model: RPNet8,
    loader: DataLoader,
    device: torch.device,
    save_predictions: str | Path | None = None,
) -> dict[str, float]:
    model.eval()
    totals = {"iou": 0.0, "detected": 0, "chars": 0, "plates": 0, "gt_chars": 0, "gt_plates": 0}
    samples = 0
    rows: list[dict[str, object]] = []
    torch.cuda.synchronize() if device.type == "cuda" else None
    started = time.perf_counter()
    for batch in tqdm(loader, desc="evaluate", leave=False, disable=True):
        image = batch["image"].to(device, non_blocking=True)
        target_box = batch["box"].to(device, non_blocking=True)
        target_plate = batch["plate"].to(device, non_blocking=True)
        pred_box, pred_logits, features = model(image)
        gt_logits = model.recognize(features, target_box)
        pred_plate = logits_to_plate(pred_logits)
        gt_plate = logits_to_plate(gt_logits)
        iou = box_iou_aligned(pred_box, target_box)
        totals["iou"] += float(iou.sum())
        totals["detected"] += int((iou >= 0.5).sum())
        totals["chars"] += int((pred_plate == target_plate).sum())
        totals["plates"] += int((pred_plate == target_plate).all(1).sum())
        totals["gt_chars"] += int((gt_plate == target_plate).sum())
        totals["gt_plates"] += int((gt_plate == target_plate).all(1).sum())
        samples += image.shape[0]
        if save_predictions is not None:
            for path, target, prediction, score in zip(batch["path"], target_plate, pred_plate, iou):
                rows.append({
                    "path": path,
                    "target": "_".join(map(str, target.tolist())),
                    "prediction": "_".join(map(str, prediction.tolist())),
                    "iou": float(score),
                    "correct": bool(torch.equal(target, prediction)),
                })
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - started
    metrics = {
        "mean_iou": totals["iou"] / samples,
        "detection_accuracy_iou50": totals["detected"] / samples,
        "character_accuracy": totals["chars"] / (samples * 8),
        "plate_accuracy": totals["plates"] / samples,
        "gt_box_character_accuracy": totals["gt_chars"] / (samples * 8),
        "gt_box_plate_accuracy": totals["gt_plates"] / samples,
        "fps": samples / elapsed,
        "samples": samples,
    }
    if save_predictions is not None:
        prediction_path = Path(save_predictions)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with prediction_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return metrics


def save_checkpoint(path: str | Path, model: nn.Module, epoch: int, metrics: dict[str, float], config: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics, "config": config}, destination)


def save_json(path: str | Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
