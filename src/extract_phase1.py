"""Extract and cache Phase 1 features, embeddings, and metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config import SETTINGS
from src.dataset import ASVspoof2019Dataset
from src.features import HandcraftedFeatureExtractor
from src.fusion import combine_features
from src.xlsr import XLSREmbedder


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev", "eval"], default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--with-xlsr", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    dataset = ASVspoof2019Dataset(
        split=args.split,
        max_samples=args.max_samples,
        balanced=args.balanced,
    )
    output_dir = SETTINGS.cache_root / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.metadata.to_csv(output_dir / "manifest.csv", index=False)

    handcrafted_extractor = HandcraftedFeatureExtractor(SETTINGS)
    xlsr = XLSREmbedder(SETTINGS) if args.with_xlsr else None
    completed = 0

    for sample in tqdm(dataset, desc=f"Extracting {args.split}"):
        output_path = output_dir / f"{sample['audio_id']}.npz"
        if output_path.exists() and not args.overwrite:
            continue
        waveform = sample["waveform"].numpy()
        handcrafted = handcrafted_extractor.extract(waveform)
        xlsr_embedding = xlsr.extract(waveform) if xlsr else None
        bundle = combine_features(handcrafted, xlsr_embedding)
        metadata = {
            "audio_id": sample["audio_id"],
            "speaker_id": sample["speaker_id"],
            "attack_id": sample["attack_id"],
            "label": int(sample["label"]),
            "label_text": sample["label_text"],
            "candidate_reference": sample["candidate_reference"],
            "source_path": sample["audio_path"],
            "phase": 1,
            "memory_updated": False,
        }
        np.savez_compressed(
            output_path,
            **bundle,
            metadata_json=np.asarray(json.dumps(metadata)),
        )
        completed += 1

    print(f"New cache files written: {completed}")
    print(f"Cache directory: {output_dir}")


if __name__ == "__main__":
    main()
