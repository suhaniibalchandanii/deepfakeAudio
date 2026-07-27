"""Final held-out evaluation and reference-count ablation for Phase 7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import PROJECT_ROOT, SETTINGS
from src.memory_dataset_phase5 import MemoryEpisodeDataset
from src.model_phase5 import Phase5MemoryAttention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "eval"), default="eval")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--references", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def equal_error_rate(labels: np.ndarray, spoof_scores: np.ndarray) -> dict:
    fpr, tpr, thresholds = roc_curve(labels, spoof_scores, pos_label=1)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return {
        "eer": float((fpr[index] + fnr[index]) / 2.0),
        "eer_threshold": float(thresholds[index]),
    }


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    spoof_scores: np.ndarray | None,
) -> dict:
    result = {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }
    if spoof_scores is None:
        result.update(
            {
                "roc_auc": None,
                "eer": None,
                "eer_percentage": None,
                "eer_threshold": None,
            }
        )
    else:
        eer = equal_error_rate(labels, spoof_scores)
        result.update(
            {
                "roc_auc": float(roc_auc_score(labels, spoof_scores)),
                "eer": eer["eer"],
                "eer_percentage": eer["eer"] * 100.0,
                "eer_threshold": eer["eer_threshold"],
            }
        )
    return result


def paired_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    samples: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    size = len(labels)
    accuracy_changes = np.empty(samples, dtype=np.float64)
    f1_changes = np.empty(samples, dtype=np.float64)
    for iteration in tqdm(range(samples), desc="paired bootstrap", leave=False):
        indices = rng.integers(0, size, size=size)
        y = labels[indices]
        base = baseline[indices]
        new = candidate[indices]
        accuracy_changes[iteration] = (
            accuracy_score(y, new) - accuracy_score(y, base)
        )
        f1_changes[iteration] = (
            f1_score(y, new, average="macro", zero_division=0)
            - f1_score(y, base, average="macro", zero_division=0)
        )

    def summary(values: np.ndarray) -> dict:
        low, high = np.percentile(values, [2.5, 97.5])
        return {
            "mean_change": float(values.mean()),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "probability_improvement": float(np.mean(values > 0)),
        }

    return {
        "iterations": samples,
        "accuracy_change": summary(accuracy_changes),
        "macro_f1_change": summary(f1_changes),
    }


def plot_confusion_matrices(
    results: dict[str, dict],
    output_path: Path,
) -> None:
    names = list(results)
    maximum = max(
        np.asarray(results[name]["confusion_matrix"]).max() for name in names
    )
    figure, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5))
    if len(names) == 1:
        axes = [axes]
    for axis, name in zip(axes, names):
        matrix = np.asarray(results[name]["confusion_matrix"])
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > maximum / 2 else "black",
                    fontsize=11,
                )
        axis.set_title(name.replace("_", " ").title())
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks([0, 1], ["Bona-fide", "Spoof"])
        axis.set_yticks([0, 1], ["Bona-fide", "Spoof"])
    figure.colorbar(image, ax=axes, fraction=0.025, pad=0.04)
    figure.suptitle("ASVspoof2019 LA: General vs Personalized Detection")
    figure.subplots_adjust(top=0.84, wspace=0.35)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_roc(
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    for name, values in scores.items():
        fpr, tpr, _ = roc_curve(labels, values)
        auc = roc_auc_score(labels, values)
        axis.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.4f})")
    axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title("ASVspoof2019 LA ROC Curve")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_reference_ablation(frame: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.8))
    axis.plot(
        frame["references"],
        frame["personalized_macro_f1"],
        marker="o",
        linewidth=2,
        label="Learned personalized",
    )
    axis.plot(
        frame["references"],
        frame["deployment_macro_f1"],
        marker="s",
        linewidth=2,
        label="Deployment gated",
    )
    axis.axhline(
        frame["general_macro_f1"].iloc[0],
        linestyle="--",
        color="black",
        label="General detector",
    )
    axis.set_xticks(frame["references"])
    axis.set_xlabel("Number of candidate references")
    axis.set_ylabel("Macro F1")
    axis.set_title("Candidate-Reference Memory Ablation")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


@torch.inference_mode()
def evaluate_reference_count(
    model: Phase5MemoryAttention,
    split: str,
    references: int,
    batch_size: int,
    calibration: dict,
    device: torch.device,
) -> pd.DataFrame:
    dataset = MemoryEpisodeDataset(split, references, SETTINGS.random_seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    rows = []
    for batch in tqdm(loader, desc=f"{split}, references={references}"):
        output = model(
            batch["general_logits"].to(device),
            batch["general_embedding"].to(device),
            batch["speaker_embedding"].to(device),
            batch["reference_embeddings"].to(device),
        )
        general_scores = torch.softmax(
            output["general_logits"], dim=-1
        )[:, 1].cpu().numpy()
        personalized_scores = torch.softmax(
            output["logits"], dim=-1
        )[:, 1].cpu().numpy()
        similarities = output["mean_similarity"].cpu().numpy()
        gates = output["gate"].cpu().numpy()

        general_predictions = (
            general_scores >= calibration["general_spoof_threshold"]
        ).astype(np.int64)
        personalized_predictions = (
            personalized_scores >= calibration["personalized_spoof_threshold"]
        ).astype(np.int64)
        deployment_predictions = np.where(
            (personalized_scores < calibration["personalized_spoof_threshold"])
            & (
                (1.0 - general_scores)
                >= calibration["minimum_general_bonafide_probability"]
            )
            & (
                similarities
                >= calibration["speaker_similarity_threshold"]
            ),
            0,
            1,
        ).astype(np.int64)

        for index in range(len(batch["label"])):
            rows.append(
                {
                    "audio_id": str(batch["audio_id"][index]),
                    "speaker_id": str(batch["speaker_id"][index]),
                    "true_label": int(batch["label"][index]),
                    "general_spoof_probability": float(general_scores[index]),
                    "personalized_spoof_probability": float(
                        personalized_scores[index]
                    ),
                    "speaker_similarity": float(similarities[index]),
                    "attention_gate": float(gates[index]),
                    "general_prediction": int(general_predictions[index]),
                    "personalized_prediction": int(
                        personalized_predictions[index]
                    ),
                    "deployment_prediction": int(
                        deployment_predictions[index]
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.split == "eval" and not (
        PROJECT_ROOT / "cache" / "phase5" / "eval_embeddings.npz"
    ).exists():
        raise FileNotFoundError("Missing cache/phase5/eval_embeddings.npz")

    calibration_path = PROJECT_ROOT / "checkpoints" / "phase6_calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("source_split") != "dev":
        raise ValueError("Calibration must originate from the dev split.")

    device = torch.device(SETTINGS.device)
    checkpoint = torch.load(
        PROJECT_ROOT / "checkpoints" / "phase5_best.pth",
        map_location=device,
    )
    model = Phase5MemoryAttention(**checkpoint["dimensions"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = PROJECT_ROOT / "outputs" / "phase7" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_rows = []
    selected_frame = None
    selected_metrics = None
    trained_references = int(checkpoint.get("references_per_query", 4))

    for references in sorted(set(args.references)):
        frame = evaluate_reference_count(
            model,
            args.split,
            references,
            args.batch_size,
            calibration,
            device,
        )
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
            labels,
            frame["deployment_prediction"].to_numpy(),
            None,
        )
        ablation_rows.append(
            {
                "references": references,
                "samples": len(frame),
                "general_accuracy": general["accuracy"],
                "general_macro_f1": general["macro_f1"],
                "personalized_accuracy": personalized["accuracy"],
                "personalized_macro_f1": personalized["macro_f1"],
                "deployment_accuracy": deployment["accuracy"],
                "deployment_macro_f1": deployment["macro_f1"],
            }
        )
        if references == trained_references:
            selected_frame = frame
            selected_metrics = {
                "general": general,
                "personalized": personalized,
                "deployment_gated": deployment,
            }

    if selected_frame is None:
        raise ValueError(
            f"--references must include the trained value {trained_references}"
        )
    labels = selected_frame["true_label"].to_numpy()
    bootstrap = paired_bootstrap(
        labels,
        selected_frame["general_prediction"].to_numpy(),
        selected_frame["personalized_prediction"].to_numpy(),
        args.bootstrap_samples,
        args.seed,
    )
    ablation = pd.DataFrame(ablation_rows)
    selected_frame.to_csv(output_dir / "predictions.csv", index=False)
    ablation.to_csv(output_dir / "reference_ablation.csv", index=False)

    final = {
        "dataset": "ASVspoof2019 LA",
        "split": args.split,
        "samples": int(len(selected_frame)),
        "trained_reference_count": trained_references,
        "calibration_source": "dev",
        "thresholds": {
            key: calibration[key]
            for key in [
                "general_spoof_threshold",
                "personalized_spoof_threshold",
                "speaker_similarity_threshold",
                "minimum_general_bonafide_probability",
            ]
        },
        "models": selected_metrics,
        "paired_bootstrap_personalized_vs_general": bootstrap,
        "excluded_samples_file": "outputs/phase7/excluded_eval_samples.csv",
        "interpretation_rule": (
            "Claim a statistically supported improvement only when the paired "
            "95% confidence interval is entirely above zero."
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )

    metric_table = pd.DataFrame(
        [
            {
                "model": name,
                **{
                    key: values[key]
                    for key in [
                        "accuracy",
                        "macro_precision",
                        "macro_recall",
                        "macro_f1",
                        "roc_auc",
                        "eer",
                    ]
                },
            }
            for name, values in selected_metrics.items()
        ]
    )
    metric_table.to_csv(output_dir / "paper_metrics.csv", index=False)
    plot_confusion_matrices(
        selected_metrics, output_dir / "confusion_matrices.png"
    )
    plot_roc(
        labels,
        {
            "General": selected_frame[
                "general_spoof_probability"
            ].to_numpy(),
            "Personalized": selected_frame[
                "personalized_spoof_probability"
            ].to_numpy(),
        },
        output_dir / "roc_curves.png",
    )
    plot_reference_ablation(
        ablation, output_dir / "reference_ablation.png"
    )

    print(json.dumps(final, indent=2))
    print(f"Paper-ready results: {output_dir}")


if __name__ == "__main__":
    main()
