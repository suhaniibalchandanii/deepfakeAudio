"""Central configuration for Phase 1."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class Settings:
    # Override with ASVSPOOF_LA_ROOT=/absolute/path/to/LA when required.
    la_root: Path = Path(
        os.getenv(
            "ASVSPOOF_LA_ROOT",
            PROJECT_ROOT / "data" / "ASVspoof2019" / "LA",
        )
    )
    cache_root: Path = Path(
        os.getenv("ASVSPOOF_CACHE_ROOT", PROJECT_ROOT / "cache" / "phase1")
    )
    sample_rate: int = 16_000
    duration_seconds: float = 4.0
    trim_silence: bool = False
    trim_top_db: int = 35
    segment_mode: str = "center"  # center, start, random
    random_seed: int = 42
    n_mfcc: int = 40
    n_lfcc: int = 40
    n_fft: int = 1024
    hop_length: int = 320
    win_length: int = 400
    n_mels: int = 80
    xlsr_model_name: str = "facebook/wav2vec2-xls-r-300m"
    xlsr_layer: int = -1
    device: str = choose_device()

    @property
    def target_samples(self) -> int:
        return int(self.sample_rate * self.duration_seconds)

    @property
    def protocol_dir(self) -> Path:
        return self.la_root / "ASVspoof2019_LA_cm_protocols"

    def audio_dir(self, split: str) -> Path:
        split = split.lower()
        names = {
            "train": "ASVspoof2019_LA_train",
            "dev": "ASVspoof2019_LA_dev",
            "eval": "ASVspoof2019_LA_eval",
        }
        if split not in names:
            raise ValueError("split must be train, dev, or eval")
        return self.la_root / names[split] / "flac"

    def protocol_path(self, split: str) -> Path:
        split = split.lower()
        names = {
            "train": "ASVspoof2019.LA.cm.train.trn.txt",
            "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
            "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
        }
        if split not in names:
            raise ValueError("split must be train, dev, or eval")
        return self.protocol_dir / names[split]


SETTINGS = Settings()
LABEL_TO_ID = {"bonafide": 0, "spoof": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
