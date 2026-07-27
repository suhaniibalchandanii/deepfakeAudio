"""Reproducible general-vs-personalized simulation on cached ASVspoof data."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm

from src.cached_dataset import CachedFeatureDataset
from src.config import PROJECT_ROOT
from src.personalized_phase4 import PersonalizedDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "eval"], default="dev")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    dataset = CachedFeatureDataset(args.split)
    if args.max_samples is not None:
        dataset.records = dataset.records[: args.max_samples]

    # First bona-fide utterance per speaker is excluded from evaluation and
    # used solely as that speaker's candidate reference.
    first_genuine: dict[str, int] = {}
    samples_by_speaker: dict[str, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        sample = dataset[index]
        speaker = str(sample["speaker_id"])
        if not speaker:
            continue
        samples_by_speaker[speaker].append(index)
        if int(sample["label"]) == 0 and speaker not in first_genuine:
            first_genuine[speaker] = index

    output_dir = PROJECT_ROOT / "outputs" / "phase4" / args.split
    memory_root = output_dir / "simulation_memory"
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = PersonalizedDetector(memory_root=memory_root)

    enrolled: set[str] = set()
    enrollment_log = []
    for speaker, index in tqdm(first_genuine.items(), desc="enroll"):
        result = detector.process_cached_sample(
            user_id=speaker,
            sample=dataset[index],
            allow_memory_update=True,
        )
        if result["memory_initialized"]:
            enrolled.add(speaker)
        enrollment_log.append(
            {
                "speaker_id": speaker,
                "audio_id": dataset[index]["audio_id"],
                "enrolled": result["memory_initialized"],
                "general_bonafide_probability": result[
                    "general_bonafide_probability"
                ],
            }
        )

    rows = []
    for speaker in tqdm(sorted(enrolled), desc="evaluate"):
        enrollment_index = first_genuine[speaker]
        for index in samples_by_speaker[speaker]:
            if index == enrollment_index:
                continue
            sample = dataset[index]
            result = detector.process_cached_sample(
                user_id=speaker,
                sample=sample,
                allow_memory_update=False,
            )
            true_label = int(sample["label"])
            rows.append(
                {
                    "speaker_id": speaker,
                    "audio_id": sample["audio_id"],
                    "true_label": true_label,
                    "general_prediction": result["general_prediction"],
                    "personalized_prediction": result["final_prediction"],
                    "general_bonafide_probability": result[
                        "general_bonafide_probability"
                    ],
                    "personalized_bonafide_probability": result[
                        "personalized_bonafide_probability"
                    ],
                    "speaker_similarity": result["speaker_similarity"],
                }
            )

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise RuntimeError("No evaluation trials were created")
    true = predictions["true_label"].to_numpy()
    general = predictions["general_prediction"].to_numpy()
    personalized = predictions["personalized_prediction"].to_numpy()
    metrics = {
        "split": args.split,
        "evaluation_samples": len(predictions),
        "enrolled_speakers": len(enrolled),
        "general_accuracy": float(accuracy_score(true, general)),
        "personalized_accuracy": float(accuracy_score(true, personalized)),
        "general_macro_f1": float(f1_score(true, general, average="macro")),
        "personalized_macro_f1": float(
            f1_score(true, personalized, average="macro")
        ),
        "general_confusion_matrix": confusion_matrix(
            true, general, labels=[0, 1]
        ).tolist(),
        "personalized_confusion_matrix": confusion_matrix(
            true, personalized, labels=[0, 1]
        ).tolist(),
    }
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(enrollment_log).to_csv(
        output_dir / "enrollment.csv", index=False
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
