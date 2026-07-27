"""Discover likely ASVspoof2021 LA audio and protocol paths on Kaggle."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/kaggle/input"))
    args = parser.parse_args()
    print("Protocol/key candidates:")
    candidates = []
    for path in args.root.rglob("*"):
        if path.is_file() and any(
            token in path.name.lower()
            for token in ("protocol", "key", "trial", "metadata")
        ):
            candidates.append(path)
    for path in candidates[:100]:
        print(path)

    audio = list(args.root.rglob("*.flac"))
    if not audio:
        audio = list(args.root.rglob("*.wav"))
    print(f"\nAudio files found: {len(audio)}")
    for path in audio[:20]:
        print(path)
    if audio:
        print(f"\nSuggested audio root: {Path(audio[0]).parent}")


if __name__ == "__main__":
    main()
