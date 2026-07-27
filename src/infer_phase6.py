"""Run calibrated learned personalization on one raw input audio file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.audio import AudioPreprocessor
from src.config import SETTINGS
from src.features import HandcraftedFeatureExtractor
from src.fusion import combine_features
from src.online_detector_phase6 import Phase6OnlineDetector
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
    parser.add_argument("--no-update", action="store_true")
    args = parser.parse_args()
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)
    detector = Phase6OnlineDetector()
    result = detector.process(
        user_id=args.user_id,
        sample=prepare_sample(args.audio),
        allow_memory_update=not args.no_update,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
