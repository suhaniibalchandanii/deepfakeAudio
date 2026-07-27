"""Evaluate classification and genuine-speaker embedding separation."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cached_dataset import CachedFeatureDataset
from src.config import PROJECT_ROOT, SETTINGS
from src.model_phase3 import Phase3SpeakerAwareDetector


def speaker_pair_statistics(
    embeddings: np.ndarray,
    speakers: list[str],
    max_pairs: int = 100_000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    count = len(speakers)
    if count < 2:
        return {}
    left = rng.integers(0, count, size=max_pairs)
    right = rng.integers(0, count, size=max_pairs)
    valid = left != right
    left, right = left[valid], right[valid]
    scores = np.sum(embeddings[left] * embeddings[right], axis=1)
    same = np.asarray([speakers[a] == speakers[b] for a, b in zip(left, right)])
    if not same.any() or same.all():
        return {}
    fpr, tpr, thresholds = roc_curve(same.astype(int), scores)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return {
        "same_speaker_cosine_mean": float(scores[same].mean()),
        "different_speaker_cosine_mean": float(scores[~same].mean()),
        "speaker_pair_eer": float((fpr[index] + fnr[index]) / 2),
        "speaker_pair_threshold": float(thresholds[index]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "eval"], default="dev")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    device = torch.device(SETTINGS.device)
    checkpoint = torch.load(
        PROJECT_ROOT / "checkpoints" / "phase3_best.pth",
        map_location=device,
    )
    dimensions = dict(checkpoint["dimensions"])
    model = Phase3SpeakerAwareDetector(**dimensions).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = CachedFeatureDataset(args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    labels, predictions, scores, ids = [], [], [], []
    genuine_embeddings, genuine_speakers = [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="evaluate"):
            output = model(
                batch["mfcc"].to(device),
                batch["lfcc"].to(device),
                batch["handcrafted"].to(device),
                batch["xlsr"].to(device),
            )
            probability = torch.softmax(output["logits"], dim=1)[:, 1]
            prediction = output["logits"].argmax(1)
            labels.extend(batch["label"].tolist())
            predictions.extend(prediction.cpu().tolist())
            scores.extend(probability.cpu().tolist())
            ids.extend(batch["audio_id"])
            mask = batch["label"] == 0
            genuine_embeddings.extend(
                output["speaker_embedding"][mask.to(device)].cpu().tolist()
            )
            genuine_speakers.extend(
                [
                    speaker
                    for speaker, keep in zip(batch["speaker_id"], mask.tolist())
                    if keep and speaker
                ]
            )

    metrics = {
        "split": args.split,
        "samples": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }
    if genuine_embeddings and len(genuine_embeddings) == len(genuine_speakers):
        metrics.update(
            speaker_pair_statistics(
                np.asarray(genuine_embeddings, dtype=np.float32),
                genuine_speakers,
            )
        )

    output_dir = PROJECT_ROOT / "outputs" / "phase3" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "audio_id": ids,
            "true_label": labels,
            "predicted_label": predictions,
            "spoof_probability": scores,
            "correct": np.asarray(labels) == np.asarray(predictions),
        }
    ).to_csv(output_dir / "predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
