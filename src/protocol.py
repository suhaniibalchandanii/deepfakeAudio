"""ASVspoof2019 LA protocol parsing and reproducible subset selection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import LABEL_TO_ID


PROTOCOL_COLUMNS = [
    "speaker_id",
    "audio_id",
    "unused",
    "attack_id",
    "label_text",
]


def read_protocol(path: Path, audio_dir: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Protocol not found: {path}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    frame = pd.read_csv(
        path,
        sep=r"\s+",
        names=PROTOCOL_COLUMNS,
        header=None,
        dtype=str,
    )
    if frame.shape[1] != 5:
        raise ValueError(f"Expected five protocol columns, found {frame.shape[1]}")
    if not frame["label_text"].isin(LABEL_TO_ID).all():
        bad = sorted(frame.loc[~frame["label_text"].isin(LABEL_TO_ID), "label_text"].unique())
        raise ValueError(f"Unknown protocol labels: {bad}")

    frame["label"] = frame["label_text"].map(LABEL_TO_ID).astype("int64")
    frame["audio_path"] = frame["audio_id"].map(
        lambda audio_id: str(audio_dir / f"{audio_id}.flac")
    )
    # Phase 1 only marks eligibility. Trust is decided by a confidence gate later.
    frame["candidate_reference"] = frame["label"].eq(0)
    return frame


def select_subset(
    frame: pd.DataFrame,
    max_samples: int | None,
    balanced: bool,
    seed: int,
) -> pd.DataFrame:
    if max_samples is None or max_samples >= len(frame):
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    if not balanced:
        return frame.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    counts = {0: (max_samples + 1) // 2, 1: max_samples // 2}
    pieces = []
    for label, requested in counts.items():
        pool = frame[frame["label"] == label]
        if len(pool) < requested:
            raise ValueError(
                f"Cannot select {requested} samples for label {label}; only {len(pool)} exist"
            )
        pieces.append(pool.sample(n=requested, random_state=seed + label))
    return (
        pd.concat(pieces, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
