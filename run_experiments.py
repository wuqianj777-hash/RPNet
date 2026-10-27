from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from rpnet.ccpd import CCPD2020Dataset
from rpnet.engine import evaluate, save_checkpoint, save_json, train_epoch
from rpnet.model import RPNet8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate CCPD2020 RPNet8 experiments")
    parser.add_argument("--config", default="configs/ccpd2020.yaml")
    parser.add_argument("--mode", choices=("all", "train", "evaluate"), default="all")
    parser.add_argument("--checkpoint", help="Checkpoint for evaluate mode")
    parser.add_argument("--epochs", type=int, help="Override joint training epochs")
    parser.add_argument("--limit", type=int, help="Limit each split for a smoke test")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(config: dict, split: str, preprocess: str, augment: bool, limit: int | None) -> DataLoader:
    dataset = CCPD2020Dataset(
        config["data_root"], split, tuple(config["image_size"]), preprocess=preprocess, augment=augment
    )
    if limit is not None:
        dataset.paths = dataset.paths[:limit]
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=split == "train",
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config["num_workers"] > 0,
    )


def load_model(config: dict, device: torch.device, checkpoint: str | Path | None = None) -> RPNet8:
    model = RPNet8(tuple(config["roi_size"])).to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    return model


def fit_baseline(config: dict, device: torch.device, limit: int | None) -> Path:
    output = Path(config["output_dir"])
    train_loader = make_loader(config, "train", "none", True, limit)
    val_loader = make_loader(config, "val", "none", False, limit)
    model = load_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler(device.type, enabled=config["amp"] and device.type == "cuda")
    best_score = -1.0
    best_path = output / "checkpoints" / "baseline_best.pt"
    history = []
    det_epochs = 0 if limit else config["det_epochs"]
    joint_epochs = config["joint_epochs"] if not limit else 1
    for epoch in range(det_epochs + joint_epochs):
        detection_only = epoch < det_epochs
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, config["box_loss_weight"], detection_only
        )
        val_metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch + 1, "phase": "detection" if detection_only else "joint", **train_metrics, **val_metrics}
        history.append(record)
        print(record)
        score = val_metrics["plate_accuracy"] + 0.1 * val_metrics["detection_accuracy_iou50"]
        if not detection_only and score > best_score:
            best_score = score
            save_checkpoint(best_path, model, epoch + 1, val_metrics, config)
    save_json(output / "metrics" / "baseline_history.json", history)
    return best_path


def finetune_clahe(config: dict, device: torch.device, baseline: Path, limit: int | None) -> Path:
    output = Path(config["output_dir"])
    train_loader = make_loader(config, "train", "clahe", False, limit)
    val_loader = make_loader(config, "val", "clahe", False, limit)
    model = load_model(config, device, baseline)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["finetune_learning_rate"], weight_decay=config["weight_decay"]
    )
    scaler = torch.amp.GradScaler(device.type, enabled=config["amp"] and device.type == "cuda")
    best_score = -1.0
    best_path = output / "checkpoints" / "clahe_finetuned_best.pt"
    history = []
    epochs = config["finetune_epochs"] if not limit else 1
    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, config["box_loss_weight"])
        val_metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch + 1, **train_metrics, **val_metrics}
        history.append(record)
        print(record)
        score = val_metrics["plate_accuracy"] + 0.1 * val_metrics["detection_accuracy_iou50"]
        if score > best_score:
            best_score = score
            save_checkpoint(best_path, model, epoch + 1, val_metrics, config)
    save_json(output / "metrics" / "clahe_finetune_history.json", history)
    return best_path


def run_evaluation(config: dict, device: torch.device, baseline: Path, finetuned: Path, limit: int | None) -> None:
    output = Path(config["output_dir"])
    experiments = [
        ("baseline", baseline, "none"),
        ("clahe_no_finetune", baseline, "clahe"),
        ("clahe_finetuned", finetuned, "clahe"),
    ]
    results = {}
    for name, checkpoint, preprocess in experiments:
        loader = make_loader(config, "test", preprocess, False, limit)
        model = load_model(config, device, checkpoint)
        metrics = evaluate(model, loader, device, output / "predictions" / f"{name}.csv")
        results[name] = metrics
        print(name, metrics)
    save_json(output / "metrics" / "test_results.json", results)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["joint_epochs"] = args.epochs
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if args.mode == "evaluate":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for evaluate mode")
        loader = make_loader(config, "test", config.get("preprocess", "none"), False, args.limit)
        metrics = evaluate(load_model(config, device, args.checkpoint), loader, device)
        print(metrics)
        return
    baseline = fit_baseline(config, device, args.limit)
    if args.mode == "train":
        return
    finetuned = finetune_clahe(config, device, baseline, args.limit)
    run_evaluation(config, device, baseline, finetuned, args.limit)


if __name__ == "__main__":
    main()
