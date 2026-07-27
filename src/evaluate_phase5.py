"""Compare the frozen general detector with learned memory personalization."""

from __future__ import annotations

import argparse
import json

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

from src.config import PROJECT_ROOT, SETTINGS
from src.memory_dataset_phase5 import MemoryEpisodeDataset
from src.model_phase5 import Phase5MemoryAttention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "eval"), default="dev")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def metrics(labels, predictions, scores):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            target_names=["bonafide", "spoof"],
            output_dict=True,
            zero_division=0,
        ),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(SETTINGS.device)
    checkpoint = torch.load(
        PROJECT_ROOT / "checkpoints" / "phase5_best.pth",
        map_location=device,
    )
    dataset = MemoryEpisodeDataset(
        args.split,
        checkpoint.get("references_per_query", 4),
        SETTINGS.random_seed,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    model = Phase5MemoryAttention(**checkpoint["dimensions"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = []
    for batch in tqdm(loader, desc=f"evaluate {args.split}"):
        output = model(
            batch["general_logits"].to(device),
            batch["general_embedding"].to(device),
            batch["speaker_embedding"].to(device),
            batch["reference_embeddings"].to(device),
        )
        general_probability = torch.softmax(
            output["general_logits"], dim=-1
        )[:, 1].cpu().numpy()
        personalized_probability = torch.softmax(
            output["logits"], dim=-1
        )[:, 1].cpu().numpy()
        general_prediction = (general_probability >= 0.5).astype(int)
        personalized_prediction = (personalized_probability >= 0.5).astype(int)
        for index in range(len(batch["label"])):
            rows.append(
                {
                    "audio_id": batch["audio_id"][index],
                    "speaker_id": batch["speaker_id"][index],
                    "true_label": int(batch["label"][index]),
                    "general_spoof_probability": float(
                        general_probability[index]
                    ),
                    "general_prediction": int(general_prediction[index]),
                    "personalized_spoof_probability": float(
                        personalized_probability[index]
                    ),
                    "personalized_prediction": int(
                        personalized_prediction[index]
                    ),
                    "speaker_similarity": float(
                        output["mean_similarity"][index].cpu()
                    ),
                    "attention_gate": float(output["gate"][index].cpu()),
                }
            )

    frame = pd.DataFrame(rows)
    labels = frame["true_label"].to_numpy()
    general = metrics(
        labels,
        frame["general_prediction"].to_numpy(),
        frame["general_spoof_probability"].to_numpy(),
    )
    personalized = metrics(
        labels,
        frame["personalized_prediction"].to_numpy(),
        frame["personalized_spoof_probability"].to_numpy(),
    )
    result = {
        "split": args.split,
        "samples": len(frame),
        "general": general,
        "personalized": personalized,
        "accuracy_change": personalized["accuracy"] - general["accuracy"],
        "macro_f1_change": personalized["macro_f1"] - general["macro_f1"],
        "note": (
            "Candidate references are separate bona-fide utterances from the "
            "same speaker. No query is used as its own reference."
        ),
    }
    output_dir = PROJECT_ROOT / "outputs" / "phase5" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
