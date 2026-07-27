"""Evaluate Phase 2 and save predictions and paper-ready metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cached_dataset import CachedFeatureDataset
from src.config import PROJECT_ROOT, SETTINGS
from src.model_phase2 import Phase2GeneralDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "eval"], default="dev")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    checkpoint_path = PROJECT_ROOT / "checkpoints" / "phase2_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    device = torch.device(SETTINGS.device)
    model = Phase2GeneralDetector(**checkpoint["dimensions"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = CachedFeatureDataset(args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    labels, predictions, probabilities, identifiers, weights = [], [], [], [], []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="evaluate"):
            output = model(
                batch["mfcc"].to(device),
                batch["lfcc"].to(device),
                batch["handcrafted"].to(device),
                batch["xlsr"].to(device),
            )
            probability = torch.softmax(output["logits"], dim=1)[:, 1]
            labels.extend(batch["label"].tolist())
            predictions.extend(output["logits"].argmax(dim=1).cpu().tolist())
            probabilities.extend(probability.cpu().tolist())
            identifiers.extend(batch["audio_id"])
            weights.extend(output["branch_weights"].cpu().tolist())

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    metrics = {
        "split": args.split,
        "samples": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            target_names=["bonafide", "spoof"],
            output_dict=True,
            zero_division=0,
        ),
    }
    output_dir = PROJECT_ROOT / "outputs" / "phase2" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    weights_array = np.asarray(weights)
    pd.DataFrame(
        {
            "audio_id": identifiers,
            "true_label": labels,
            "predicted_label": predictions,
            "spoof_probability": probabilities,
            "correct": np.asarray(labels) == np.asarray(predictions),
            "mfcc_weight": weights_array[:, 0],
            "lfcc_weight": weights_array[:, 1],
            "handcrafted_weight": weights_array[:, 2],
            "xlsr_weight": weights_array[:, 3],
        }
    ).to_csv(output_dir / "predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
