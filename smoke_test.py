"""Fast smoke test using synthetic audio and a temporary ASVspoof protocol."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio import AudioPreprocessor
from src.config import SETTINGS
from src.features import HandcraftedFeatureExtractor
from src.fusion import combine_features
from src.protocol import read_protocol, select_subset


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        audio_dir = root / "flac"
        audio_dir.mkdir()
        seconds = 3.2
        time = np.arange(int(SETTINGS.sample_rate * seconds)) / SETTINGS.sample_rate
        waveform = (0.25 * np.sin(2 * np.pi * 180 * time)).astype(np.float32)

        rows = [
            ("LA_0001", "LA_T_0000001", "-", "-", "bonafide"),
            ("LA_0002", "LA_T_0000002", "-", "A01", "spoof"),
        ]
        for _, audio_id, *_ in rows:
            sf.write(audio_dir / f"{audio_id}.flac", waveform, SETTINGS.sample_rate)
        protocol = root / "protocol.txt"
        protocol.write_text(
            "\n".join(" ".join(row) for row in rows) + "\n",
            encoding="utf-8",
        )

        frame = read_protocol(protocol, audio_dir)
        subset = select_subset(frame, max_samples=2, balanced=True, seed=42)
        assert subset["label"].tolist().count(0) == 1
        assert subset["label"].tolist().count(1) == 1

        processed = AudioPreprocessor(SETTINGS)(subset.iloc[0]["audio_path"])
        assert processed.shape == (SETTINGS.target_samples,)
        handcrafted = HandcraftedFeatureExtractor(SETTINGS).extract(processed)
        bundle = combine_features(handcrafted, xlsr_embedding=None)

        assert bundle["mfcc"].shape[0] == SETTINGS.n_mfcc * 3
        assert bundle["lfcc"].shape[0] == SETTINGS.n_lfcc
        assert bundle["hybrid_vector"].ndim == 1
        assert np.isfinite(bundle["hybrid_vector"]).all()
        print("Phase 1 smoke test passed.")
        for key, value in bundle.items():
            print(f"{key:20s} {value.shape}")
        print("XLS-R intentionally skipped; use --with-xlsr for real extraction.")


if __name__ == "__main__":
    main()
