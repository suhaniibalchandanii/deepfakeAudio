"""Cross-dataset ASVspoof2021 LA evaluation without 2021 training."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import PROJECT_ROOT, SETTINGS
from src.evaluate_phase7 import calculate_metrics
from src.model_phase5 import Phase5MemoryAttention
from src.phase8_data import Phase8MemoryDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--references", type=int, nargs="+", default=[1, 2, 4, 8])
    return parser.parse_args()


@torch.inference_mode()
def evaluate(model, dataset, loader, calibration, device):
    rows = []
    for batch in tqdm(loader, desc="ASVspoof2021 cross-dataset evaluation"):
        output = model(
            batch["general_logits"].to(device),
            batch["general_embedding"].to(device),
            batch["speaker_embedding"].to(device),
            batch["reference_embeddings"].to(device),
        )
        general_scores = torch.softmax(
            output["general_logits"], dim=-1
        )[:, 1].cpu().numpy()
        personal_scores = torch.softmax(
            output["logits"], dim=-1
        )[:, 1].cpu().numpy()
        similarities = output["mean_similarity"].cpu().numpy()
        general_predictions = (
            general_scores >= calibration["general_spoof_threshold"]
        ).astype(int)
        personal_predictions = (
            personal_scores >= calibration["personalized_spoof_threshold"]
        ).astype(int)
        deployment_predictions = np.where(
            (personal_scores < calibration["personalized_spoof_threshold"])
            & (
                1.0 - general_scores
                >= calibration["minimum_general_bonafide_probability"]
            )
            & (
                similarities
                >= calibration["speaker_similarity_threshold"]
            ),
            0,
            1,
        ).astype(int)
        for index in range(len(batch["label"])):
            rows.append(
                {
                    "audio_id": str(batch["audio_id"][index]),
                    "speaker_id": str(batch["speaker_id"][index]),
                    "true_label": int(batch["label"][index]),
                    "general_spoof_probability": float(general_scores[index]),
                    "personalized_spoof_probability": float(
                        personal_scores[index]
                    ),
                    "speaker_similarity": float(similarities[index]),
                    "general_prediction": int(general_predictions[index]),
                    "personalized_prediction": int(
                        personal_predictions[index]
                    ),
                    "deployment_prediction": int(
                        deployment_predictions[index]
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    embedding_path = (
        PROJECT_ROOT / "cache" / "phase8_2021" / "la_eval_embeddings.npz"
    )
    calibration = json.loads(
        (
            PROJECT_ROOT / "checkpoints" / "phase6_calibration.json"
        ).read_text(encoding="utf-8")
    )
    if calibration.get("source_split") != "dev":
        raise ValueError("Calibration must come from ASVspoof2019 dev.")
    device = torch.device(SETTINGS.device)
    checkpoint = torch.load(
        PROJECT_ROOT / "checkpoints" / "phase5_best.pth",
        map_location=device,
    )
    model = Phase5MemoryAttention(**checkpoint["dimensions"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    output_dir = PROJECT_ROOT / "outputs" / "phase8" / "asvspoof2021_la"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for references in sorted(set(args.references)):
        dataset = Phase8MemoryDataset(embedding_path, references, 42)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        frame = evaluate(model, dataset, loader, calibration, device)
        labels = frame["true_label"].to_numpy()
        general = calculate_metrics(
            labels,
            frame["general_prediction"].to_numpy(),
            frame["general_spoof_probability"].to_numpy(),
        )
        personalized = calculate_metrics(
            labels,
            frame["personalized_prediction"].to_numpy(),
            frame["personalized_spoof_probability"].to_numpy(),
        )
        deployment = calculate_metrics(
            labels, frame["deployment_prediction"].to_numpy(), None
        )
        rows.append(
            {
                "references": references,
                "samples": len(frame),
                "general_accuracy": general["accuracy"],
                "general_macro_f1": general["macro_f1"],
                "general_auc": general["roc_auc"],
                "general_eer": general["eer"],
                "personalized_accuracy": personalized["accuracy"],
                "personalized_macro_f1": personalized["macro_f1"],
                "personalized_auc": personalized["roc_auc"],
                "personalized_eer": personalized["eer"],
                "deployment_accuracy": deployment["accuracy"],
                "deployment_macro_f1": deployment["macro_f1"],
            }
        )
        if references == int(checkpoint.get("references_per_query", 4)):
            frame.to_csv(output_dir / "predictions.csv", index=False)
            details = {
                "dataset": "ASVspoof2021 LA",
                "protocol": "cross-dataset; no ASVspoof2021 training",
                "references": references,
                "general": general,
                "personalized": personalized,
                "deployment_gated": deployment,
                "calibration_source": "ASVspoof2019 LA dev",
            }
            (output_dir / "metrics.json").write_text(
                json.dumps(details, indent=2), encoding="utf-8"
            )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "reference_ablation.csv", index=False)
    print(table.to_string(index=False))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
