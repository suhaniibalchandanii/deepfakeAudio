"""Handcrafted acoustic, spectral, prosodic, and physics-inspired features."""

from __future__ import annotations

import warnings

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.fft import dct

from src.config import Settings


def _summary(values: np.ndarray, size: int = 4) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros(size, dtype=np.float32)
    stats = [values.mean(), values.std(), np.median(values), np.ptp(values)]
    return np.asarray(stats[:size], dtype=np.float32)


class HandcraftedFeatureExtractor:
    def __init__(self, settings: Settings) -> None:
        self.s = settings

    def mfcc_sequence(self, waveform: np.ndarray) -> np.ndarray:
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=self.s.sample_rate,
            n_mfcc=self.s.n_mfcc,
            n_fft=self.s.n_fft,
            hop_length=self.s.hop_length,
            win_length=self.s.win_length,
            n_mels=self.s.n_mels,
        )
        delta = librosa.feature.delta(mfcc, mode="nearest")
        delta2 = librosa.feature.delta(mfcc, order=2, mode="nearest")
        return np.concatenate([mfcc, delta, delta2], axis=0).astype(np.float32)

    def lfcc_sequence(self, waveform: np.ndarray) -> np.ndarray:
        """LFCC via a linear-frequency filterbank followed by log and DCT."""
        spectrum = np.abs(
            librosa.stft(
                waveform,
                n_fft=self.s.n_fft,
                hop_length=self.s.hop_length,
                win_length=self.s.win_length,
            )
        ) ** 2
        bins = spectrum.shape[0]
        edges = np.linspace(0, bins - 1, self.s.n_lfcc + 2).astype(int)
        filters = np.zeros((self.s.n_lfcc, bins), dtype=np.float32)
        for index in range(self.s.n_lfcc):
            left, center, right = edges[index : index + 3]
            if center > left:
                filters[index, left:center] = np.linspace(0, 1, center - left, endpoint=False)
            if right > center:
                filters[index, center:right] = np.linspace(1, 0, right - center, endpoint=False)
        log_energy = np.log(np.maximum(filters @ spectrum, 1e-10))
        return dct(log_energy, type=2, axis=0, norm="ortho")[: self.s.n_lfcc].astype(
            np.float32
        )

    def spectral_vector(self, waveform: np.ndarray) -> np.ndarray:
        kwargs = dict(
            y=waveform,
            sr=self.s.sample_rate,
            n_fft=self.s.n_fft,
            hop_length=self.s.hop_length,
        )
        centroid = librosa.feature.spectral_centroid(**kwargs)
        bandwidth = librosa.feature.spectral_bandwidth(**kwargs)
        rolloff = librosa.feature.spectral_rolloff(**kwargs)
        contrast = librosa.feature.spectral_contrast(**kwargs)
        flatness = librosa.feature.spectral_flatness(
            y=waveform,
            n_fft=self.s.n_fft,
            hop_length=self.s.hop_length,
        )
        chroma = librosa.feature.chroma_stft(**kwargs)
        rms = librosa.feature.rms(
            y=waveform,
            frame_length=self.s.n_fft,
            hop_length=self.s.hop_length,
        )
        zcr = librosa.feature.zero_crossing_rate(
            waveform,
            frame_length=self.s.n_fft,
            hop_length=self.s.hop_length,
        )
        pieces = [
            _summary(centroid),
            _summary(bandwidth),
            _summary(rolloff),
            _summary(contrast),
            _summary(flatness),
            chroma.mean(axis=1).astype(np.float32),
            _summary(rms),
            _summary(zcr),
        ]
        return np.concatenate(pieces).astype(np.float32)

    def prosodic_vector(self, waveform: np.ndarray) -> np.ndarray:
        f0, voiced, _ = librosa.pyin(
            waveform,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=self.s.sample_rate,
            frame_length=self.s.n_fft,
            hop_length=self.s.hop_length,
        )
        voiced = np.asarray(voiced, dtype=bool)
        valid_f0 = f0[np.isfinite(f0) & voiced]
        rms = librosa.feature.rms(
            y=waveform,
            frame_length=self.s.n_fft,
            hop_length=self.s.hop_length,
        ).squeeze()
        silence_threshold = max(float(np.percentile(rms, 20)), 1e-5)
        silent = rms <= silence_threshold
        transitions = int(np.sum(np.diff(silent.astype(np.int8)) != 0))
        voiced_ratio = float(voiced.mean()) if voiced.size else 0.0
        return np.concatenate(
            [
                _summary(valid_f0),
                _summary(rms),
                np.asarray(
                    [
                        voiced_ratio,
                        float(silent.mean()),
                        float(transitions),
                        float(np.std(np.diff(valid_f0))) if valid_f0.size > 1 else 0.0,
                    ],
                    dtype=np.float32,
                ),
            ]
        )

    def physics_vector(self, waveform: np.ndarray) -> np.ndarray:
        """Praat-derived formants, HNR, jitter, and shimmer with safe fallbacks."""
        try:
            sound = parselmouth.Sound(waveform, sampling_frequency=self.s.sample_rate)
            pitch = call(sound, "To Pitch", 0.0, 75.0, 600.0)
            point = call(sound, "To PointProcess (periodic, cc)", 75.0, 600.0)
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
            formant = call(sound, "To Formant (burg)", 0.0, 5, 5500.0, 0.025, 50.0)

            times = np.arange(0.05, max(sound.duration - 0.05, 0.06), 0.01)
            formants = []
            for number in (1, 2, 3):
                values = [call(formant, "Get value at time", number, t, "Hertz", "Linear") for t in times]
                formants.append(_summary(np.asarray(values)))

            hnr = np.asarray(harmonicity.values).squeeze()
            pitch_values = np.asarray(pitch.selected_array["frequency"])
            jitter = float(call(point, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
            shimmer = float(
                call([sound, point], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            )
            return np.concatenate(
                formants
                + [
                    _summary(hnr),
                    _summary(pitch_values[pitch_values > 0]),
                    np.asarray([jitter, shimmer], dtype=np.float32),
                ]
            ).astype(np.float32)
        except Exception as error:
            warnings.warn(f"Praat feature extraction failed; using zeros: {error}")
            return np.zeros(22, dtype=np.float32)

    def extract(self, waveform: np.ndarray) -> dict[str, np.ndarray]:
        mfcc = self.mfcc_sequence(waveform)
        lfcc = self.lfcc_sequence(waveform)
        spectral = self.spectral_vector(waveform)
        prosodic = self.prosodic_vector(waveform)
        physics = self.physics_vector(waveform)
        global_vector = np.concatenate([spectral, prosodic, physics]).astype(np.float32)
        return {
            "mfcc": mfcc,
            "lfcc": lfcc,
            "spectral": spectral,
            "prosodic": prosodic,
            "physics": physics,
            "handcrafted_global": global_vector,
        }


# CQCC is deliberately deferred. A true CQCC requires constant-Q analysis,
# uniform resampling in the log-frequency domain, and a cepstral transform.
# Calling ordinary chroma/CQT magnitudes "CQCC" would be misleading.
