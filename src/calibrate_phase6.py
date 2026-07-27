"""Calibrate deployable decision thresholds using Phase 5 dev predictions only."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.config import PROJECT_ROOT


def best_spoof_threshold(labels: np.ndarray, scores: np.ndarray) -> dict:
    best = {"threshold": 0.5, "macro_f1": -1.0}
    for threshold in np.linspace(0.01, 0.99, 981):
        prediction = (scores >= threshold).astype(np.int64)
        score = f1_score(labels, prediction, average="macro")
        if score > best["macro_f1"]:
            best = {"threshold": float(threshold), "macro_f1": float(score)}
    return best


def best_similarity_threshold(labels: np.ndarray, similarities: np.ndarray) -> dict:
    # High similarity is speaker-consistent/bona-fide, hence prediction 0.
    best = {"threshold": 0.40, "balanced_accuracy": -1.0}
    for threshold in np.linspace(-1.0, 1.0, 1001):
        prediction = (similarities < threshold).astype(np.int64)
        score = balanced_accuracy_score(labels, prediction)
        if score > best["balanced_accuracy"]:
            best = {
                "threshold": float(threshold),
                "balanced_accuracy": float(score),
            }
    return best


def main() -> None:
    predictions_path = (
        PROJECT_ROOT / "outputs" / "phase5" / "dev" / "predictions.csv"
    )
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing {predictions_path}\n"
            "Run: python -m src.evaluate_phase5 --split dev"
        )
    frame = pd.read_csv(predictions_path)
    labels = frame["true_label"].to_numpy(dtype=np.int64)
    general = best_spoof_threshold(
        labels, frame["general_spoof_probability"].to_numpy()
    )
    personalized = best_spoof_threshold(
        labels, frame["personalized_spoof_probability"].to_numpy()
    )
    similarity = best_similarity_threshold(
        labels, frame["speaker_similarity"].to_numpy()
    )
    # Calibration may find a lower threshold that improves aggregate metrics,
    # but online memory protection must remain conservative.
    safe_similarity_threshold = max(0.40, similarity["threshold"])
    calibration = {
        "source_split": "dev",
        "general_spoof_threshold": general["threshold"],
        "personalized_spoof_threshold": personalized["threshold"],
        "speaker_similarity_threshold": safe_similarity_threshold,
        "initial_enrollment_bonafide_probability": 0.98,
        "minimum_general_bonafide_probability": 0.90,
        "memory_update_general_bonafide_probability": 0.98,
        "memory_update_personalized_bonafide_probability": 0.97,
        "calibration_metrics": {
            "general_macro_f1": general["macro_f1"],
            "personalized_macro_f1": personalized["macro_f1"],
            "similarity_balanced_accuracy": similarity["balanced_accuracy"],
            "raw_calibrated_similarity_threshold": similarity["threshold"],
        },
        "warning": (
            "Thresholds were selected only on dev. Report final performance "
            "on eval without recalibrating."
        ),
    }
    path = PROJECT_ROOT / "checkpoints" / "phase6_calibration.json"
    path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(json.dumps(calibration, indent=2))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
