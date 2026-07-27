"""Audio decoding, resampling, normalisation, trimming, and segmentation."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from src.config import Settings


class AudioPreprocessor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.rng = np.random.default_rng(settings.random_seed)

    def load(self, path: str | Path) -> tuple[np.ndarray, int]:
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
        return waveform, int(sample_rate)

    def resample(self, waveform: np.ndarray, source_rate: int) -> np.ndarray:
        target_rate = self.settings.sample_rate
        if source_rate == target_rate:
            return waveform.astype(np.float32, copy=False)
        divisor = int(np.gcd(source_rate, target_rate))
        return resample_poly(
            waveform,
            up=target_rate // divisor,
            down=source_rate // divisor,
        ).astype(np.float32)

    def trim(self, waveform: np.ndarray) -> np.ndarray:
        if not self.settings.trim_silence or not np.any(waveform):
            return waveform
        trimmed, _ = librosa.effects.trim(
            waveform,
            top_db=self.settings.trim_top_db,
        )
        return trimmed if trimmed.size else waveform

    @staticmethod
    def peak_normalize(waveform: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        return waveform / peak if peak > 1e-8 else waveform

    def segment(self, waveform: np.ndarray) -> np.ndarray:
        target = self.settings.target_samples
        if waveform.size < target:
            return np.pad(waveform, (0, target - waveform.size))
        if waveform.size == target:
            return waveform

        maximum_start = waveform.size - target
        mode = self.settings.segment_mode
        if mode == "start":
            start = 0
        elif mode == "center":
            start = maximum_start // 2
        elif mode == "random":
            start = int(self.rng.integers(0, maximum_start + 1))
        else:
            raise ValueError("segment_mode must be start, center, or random")
        return waveform[start : start + target]

    def __call__(self, path: str | Path) -> np.ndarray:
        waveform, source_rate = self.load(path)
        waveform = self.resample(waveform, source_rate)
        waveform = self.trim(waveform)
        waveform = self.peak_normalize(waveform)
        waveform = self.segment(waveform)
        return np.nan_to_num(waveform).astype(np.float32)
