"""Process one raw input; it may become the user's initial reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.audio import AudioPreprocessor
from src.config import SETTINGS
from src.features import HandcraftedFeatureExtractor
from src.fusion import combine_features
from src.personalized_phase4 import PersonalizedDetector
from src.xlsr import XLSREmbedder


def prepare_sample(audio_path: Path) -> dict[str, object]:
    waveform = AudioPreprocessor(SETTINGS)(audio_path)
    handcrafted = HandcraftedFeatureExtractor(SETTINGS).extract(waveform)
    xlsr = XLSREmbedder(SETTINGS).extract(waveform)
    bundle = combine_features(handcrafted, xlsr)
    return {
        "audio_id": audio_path.stem,
        "mfcc": torch.from_numpy(bundle["mfcc"]),
        "lfcc": torch.from_numpy(bundle["lfcc"]),
        "handcrafted": torch.from_numpy(bundle["handcrafted_global"]),
        "xlsr": torch.from_numpy(bundle["xlsr"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Make a decision without changing user memory",
    )
    args = parser.parse_args()
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)
    sample = prepare_sample(args.audio)
    detector = PersonalizedDetector()
    result = detector.process_cached_sample(
        user_id=args.user_id,
        sample=sample,
        allow_memory_update=not args.no_update,
    )
    # Remove large arrays before printing.
    result.pop("speaker_embedding", None)
    result.pop("hybrid_embedding", None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
