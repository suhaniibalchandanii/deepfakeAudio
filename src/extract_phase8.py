"""Resumably extract ASVspoof2021 LA features on Kaggle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.audio import AudioPreprocessor
from src.config import PROJECT_ROOT, SETTINGS
from src.features import HandcraftedFeatureExtractor
from src.fusion import combine_features
from src.phase8_data import (
    build_audio_index,
    parse_asvspoof2021_la_protocol,
    select_reproducible_subset,
)
from src.xlsr import XLSREmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / "cache" / "phase8_2021" / "la_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = parse_asvspoof2021_la_protocol(args.protocol)
    selected = select_reproducible_subset(
        frame, args.max_samples, args.balanced, args.seed
    )
    selected.to_csv(output_dir / "selected_manifest.csv", index=False)
    print("Selected label counts:")
    print(selected["label_text"].value_counts())

    audio_index = build_audio_index(args.audio_root)
    preprocessor = AudioPreprocessor(SETTINGS)
    handcrafted_extractor = HandcraftedFeatureExtractor(SETTINGS)
    xlsr = XLSREmbedder(SETTINGS)
    failures = []

    for row in tqdm(
        list(selected.itertuples(index=False)), desc="ASVspoof2021 features"
    ):
        output_path = output_dir / f"{row.audio_id}.npz"
        if output_path.exists() and not args.overwrite:
            continue
        audio_path = audio_index.get(str(row.audio_id))
        if audio_path is None:
            failures.append(
                {"audio_id": row.audio_id, "reason": "audio_not_found"}
            )
            continue
        try:
            waveform = preprocessor(audio_path)
            handcrafted = handcrafted_extractor.extract(waveform)
            xlsr_embedding = xlsr.extract(waveform)
            bundle = combine_features(handcrafted, xlsr_embedding)
            arrays_to_check = (
                bundle["mfcc"],
                bundle["lfcc"],
                bundle["handcrafted_global"],
                bundle["xlsr"],
            )
            if not all(np.isfinite(array).all() for array in arrays_to_check):
                raise ValueError("non-finite extracted feature")
            metadata = {
                "dataset": "ASVspoof2021_LA",
                "speaker_id": str(row.speaker_id),
                "audio_id": str(row.audio_id),
                "label": int(row.label),
                "label_text": str(row.label_text),
                "attack_id": str(row.attack_id),
                "codec": str(row.codec),
                "transmission": str(row.transmission),
                "candidate_reference": int(row.label) == 0,
            }
            np.savez_compressed(
                output_path,
                **bundle,
                metadata_json=np.asarray(json.dumps(metadata)),
            )
        except Exception as error:
            failures.append(
                {"audio_id": row.audio_id, "reason": repr(error)}
            )

    completed = selected[
        selected["audio_id"].map(
            lambda audio_id: (output_dir / f"{audio_id}.npz").exists()
        )
    ].copy()
    completed.to_csv(output_dir / "manifest.csv", index=False)
    failure_path = output_dir / "failures.json"
    failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"Completed caches: {len(completed)}")
    print(f"Failures this run: {len(failures)}")
    print(f"Cache: {output_dir}")


if __name__ == "__main__":
    main()
