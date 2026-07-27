"""Fine-tune Phase 2 with a genuine-speaker contrastive objective."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cached_dataset import CachedFeatureDataset, infer_feature_dimensions
from src.config import PROJECT_ROOT, SETTINGS
from src.contrastive_sampler import SpeakerContrastiveBatchSampler
from src.losses_phase3 import Phase3JointLoss
from src.model_phase3 import (
    Phase3SpeakerAwareDetector,
    load_phase2_weights,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--contrastive-weight", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def forward_batch(model, batch, device):
    return model(
        batch["mfcc"].to(device),
        batch["lfcc"].to(device),
        batch["handcrafted"].to(device),
        batch["xlsr"].to(device),
    )


def train_epoch(model, loader, loss_function, optimizer, device):
    model.train()
    totals = {"total": 0.0, "classification": 0.0, "contrastive": 0.0}
    sample_count = 0
    for batch in tqdm(loader, desc="train", leave=False):
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = forward_batch(model, batch, device)
        losses = loss_function(output, labels, list(batch["speaker_id"]))
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        size = labels.size(0)
        sample_count += size
        for key in totals:
            totals[key] += float(losses[key].item()) * size
    return {key: value / sample_count for key, value in totals.items()}


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    labels, predictions, scores = [], [], []
    for batch in tqdm(loader, desc="dev", leave=False):
        output = forward_batch(model, batch, device)
        probability = torch.softmax(output["logits"], dim=1)[:, 1]
        labels.extend(batch["label"].tolist())
        predictions.extend(output["logits"].argmax(1).cpu().tolist())
        scores.extend(probability.cpu().tolist())
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def main() -> None:
    args = arguments()
    seed_all(SETTINGS.random_seed)
    device = torch.device(SETTINGS.device)
    train_dataset = CachedFeatureDataset("train")
    dev_dataset = CachedFeatureDataset("dev")
    dimensions = infer_feature_dimensions(train_dataset)

    sampler = SpeakerContrastiveBatchSampler(
        train_dataset,
        batch_size=args.batch_size,
        seed=SETTINGS.random_seed,
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=sampler, num_workers=0
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = Phase3SpeakerAwareDetector(**dimensions).to(device)
    phase2_path = PROJECT_ROOT / "checkpoints" / "phase2_best.pth"
    latest_path = PROJECT_ROOT / "checkpoints" / "phase3_latest.pth"
    best_path = PROJECT_ROOT / "checkpoints" / "phase3_best.pth"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    loss_function = Phase3JointLoss(
        contrastive_weight=args.contrastive_weight,
        temperature=args.temperature,
    )
    start_epoch, best_f1, stale, history = 1, -1.0, 0, []

    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint["best_f1"]
        history = checkpoint.get("history", [])
        print(f"Resuming Phase 3 at epoch {start_epoch}")
    else:
        if not phase2_path.exists():
            raise FileNotFoundError(f"Phase 2 checkpoint not found: {phase2_path}")
        phase2_checkpoint = torch.load(phase2_path, map_location=device)
        missing, unexpected = load_phase2_weights(model, phase2_checkpoint)
        print("Loaded Phase 2 weights.")
        print("New Phase 3 parameters:", missing)
        print("Unexpected keys:", unexpected)

    output_dir = PROJECT_ROOT / "outputs" / "phase3"
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        sampler.set_epoch(epoch)
        train_metrics = train_epoch(
            model, train_loader, loss_function, optimizer, device
        )
        dev_metrics = validate(model, dev_loader, device)
        scheduler.step(dev_metrics["macro_f1"])
        record = {"epoch": epoch, "train": train_metrics, "dev": dev_metrics}
        history.append(record)
        print(
            f"Epoch {epoch:02d} | total={train_metrics['total']:.4f} "
            f"CE={train_metrics['classification']:.4f} "
            f"SupCon={train_metrics['contrastive']:.4f} | "
            f"dev acc={dev_metrics['accuracy']:.4f} "
            f"F1={dev_metrics['macro_f1']:.4f} "
            f"AUC={dev_metrics['roc_auc']:.4f}"
        )
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "dimensions": model.dimensions,
            "best_f1": best_f1,
            "history": history,
        }
        torch.save(state, latest_path)
        if dev_metrics["macro_f1"] > best_f1:
            best_f1, stale = dev_metrics["macro_f1"], 0
            state["best_f1"] = best_f1
            torch.save(state, best_path)
            print(f"Saved best checkpoint: {best_path}")
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping.")
                break
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
