"""PyTorch dataset returning processed waveform and reference-ready metadata."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from src.audio import AudioPreprocessor
from src.config import SETTINGS, Settings
from src.protocol import read_protocol, select_subset


class ASVspoof2019Dataset(Dataset):
    def __init__(
        self,
        split: str,
        max_samples: int | None = None,
        balanced: bool = False,
        settings: Settings = SETTINGS,
    ) -> None:
        self.split = split.lower()
        self.settings = settings
        frame = read_protocol(
            settings.protocol_path(self.split),
            settings.audio_dir(self.split),
        )
        self.metadata = select_subset(
            frame,
            max_samples=max_samples,
            balanced=balanced,
            seed=settings.random_seed,
        )
        self.preprocess = AudioPreprocessor(settings)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.metadata.iloc[index]
        waveform = self.preprocess(row["audio_path"])
        return {
            "waveform": torch.from_numpy(waveform),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "audio_id": str(row["audio_id"]),
            "speaker_id": str(row["speaker_id"]),
            "attack_id": str(row["attack_id"]),
            "label_text": str(row["label_text"]),
            "candidate_reference": bool(row["candidate_reference"]),
            "audio_path": str(row["audio_path"]),
        }
