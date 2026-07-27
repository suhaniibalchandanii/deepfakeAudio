"""Train the Phase 2 general detector from offline Phase 1 caches."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.cached_dataset import CachedFeatureDataset, infer_feature_dimensions
from src.config import PROJECT_ROOT, SETTINGS
from src.model_phase2 import Phase2GeneralDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def balanced_sampler(dataset: CachedFeatureDataset) -> WeightedRandomSampler:
    labels = np.asarray([record["label"] for record in dataset.records])
    counts = np.bincount(labels, minlength=2)
    weights = 1.0 / np.maximum(counts, 1)
    sample_weights = torch.as_tensor(weights[labels], dtype=torch.double)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def move_batch(batch: dict[str, object], device: torch.device):
    return (
        batch["mfcc"].to(device),
        batch["lfcc"].to(device),
        batch["handcrafted"].to(device),
        batch["xlsr"].to(device),
        batch["label"].to(device),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses, labels, predictions, scores = [], [], [], []

    for batch in tqdm(loader, leave=False, desc="train" if training else "dev"):
        mfcc, lfcc, handcrafted, xlsr, target = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(mfcc, lfcc, handcrafted, xlsr)
            loss = criterion(output["logits"], target)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        probability = torch.softmax(output["logits"], dim=1)[:, 1]
        prediction = output["logits"].argmax(dim=1)
        losses.append(loss.item() * target.size(0))
        labels.extend(target.detach().cpu().tolist())
        predictions.extend(prediction.detach().cpu().tolist())
        scores.extend(probability.detach().cpu().tolist())

    metrics = {
        "loss": float(sum(losses) / len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else float("nan")
    )
    return metrics


def main() -> None:
    args = parse_args()
    seed_everything(SETTINGS.random_seed)
    device = torch.device(SETTINGS.device)
    train_dataset = CachedFeatureDataset("train")
    dev_dataset = CachedFeatureDataset("dev")
    dimensions = infer_feature_dimensions(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=balanced_sampler(train_dataset),
        num_workers=0,
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = Phase2GeneralDetector(**dimensions).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    output_dir = PROJECT_ROOT / "outputs" / "phase2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "phase2_latest.pth"
    best_path = checkpoint_dir / "phase2_best.pth"
    start_epoch, best_loss, stale_epochs, history = 1, float("inf"), 0, []

    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        history = checkpoint.get("history", [])
        print(f"Resuming at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        dev_metrics = run_epoch(model, dev_loader, criterion, device, None)
        scheduler.step(dev_metrics["loss"])
        record = {"epoch": epoch, "train": train_metrics, "dev": dev_metrics}
        history.append(record)
        print(
            f"Epoch {epoch:02d} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"F1={train_metrics['macro_f1']:.4f} | "
            f"dev loss={dev_metrics['loss']:.4f} "
            f"acc={dev_metrics['accuracy']:.4f} "
            f"F1={dev_metrics['macro_f1']:.4f} "
            f"AUC={dev_metrics['roc_auc']:.4f}"
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "dimensions": model.dimensions,
            "best_loss": best_loss,
            "history": history,
        }
        torch.save(state, latest_path)
        if dev_metrics["loss"] < best_loss:
            best_loss, stale_epochs = dev_metrics["loss"], 0
            state["best_loss"] = best_loss
            torch.save(state, best_path)
            print(f"Saved best checkpoint: {best_path}")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping.")
                break

        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
