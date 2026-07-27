"""Train learned memory attention while preserving the Phase 2 baseline."""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import PROJECT_ROOT, SETTINGS
from src.memory_dataset_phase5 import MemoryEpisodeDataset
from src.model_phase5 import Phase5MemoryAttention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--references", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def forward(model, batch, device):
    return model(
        batch["general_logits"].to(device),
        batch["general_embedding"].to(device),
        batch["speaker_embedding"].to(device),
        batch["reference_embeddings"].to(device),
    )


def class_weights(dataset, device):
    labels = dataset.labels[dataset.query_indices]
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(2.0 * counts, 1.0)
    return torch.tensor(weights, device=device)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_samples = 0.0, 0
    for batch in tqdm(loader, desc="train", leave=False):
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = forward(model, batch, device)
        classification = criterion(output["logits"], labels)
        correction_penalty = output["correction"].pow(2).mean()
        loss = classification + 1e-4 * correction_penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        total_samples += labels.size(0)
    return total_loss / total_samples


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    labels, predictions, scores = [], [], []
    for batch in tqdm(loader, desc="dev", leave=False):
        output = forward(model, batch, device)
        probability = torch.softmax(output["logits"], dim=-1)[:, 1]
        labels.extend(batch["label"].tolist())
        predictions.extend(output["logits"].argmax(dim=-1).cpu().tolist())
        scores.extend(probability.cpu().tolist())
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def main() -> None:
    args = parse_args()
    random.seed(SETTINGS.random_seed)
    np.random.seed(SETTINGS.random_seed)
    torch.manual_seed(SETTINGS.random_seed)
    device = torch.device(SETTINGS.device)

    train_dataset = MemoryEpisodeDataset(
        "train", args.references, SETTINGS.random_seed
    )
    dev_dataset = MemoryEpisodeDataset(
        "dev", args.references, SETTINGS.random_seed
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    model = Phase5MemoryAttention().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights(train_dataset, device)
    )
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "phase5_latest.pth"
    best_path = checkpoint_dir / "phase5_best.pth"
    output_dir = PROJECT_ROOT / "outputs" / "phase5"
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, best_f1, stale, history = 1, -1.0, 0, []

    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint["best_f1"]
        history = checkpoint.get("history", [])

    for epoch in range(start_epoch, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        metrics = validate(model, dev_loader, device)
        scheduler.step(metrics["macro_f1"])
        record = {"epoch": epoch, "train_loss": loss, "dev": metrics}
        history.append(record)
        print(
            f"Epoch {epoch:02d} | loss={loss:.4f} | "
            f"dev accuracy={metrics['accuracy']:.4f} "
            f"F1={metrics['macro_f1']:.4f} AUC={metrics['roc_auc']:.4f}"
        )
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "dimensions": model.dimensions,
            "references_per_query": args.references,
            "best_f1": best_f1,
            "history": history,
        }
        torch.save(state, latest_path)
        if metrics["macro_f1"] > best_f1:
            best_f1, stale = metrics["macro_f1"], 0
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
