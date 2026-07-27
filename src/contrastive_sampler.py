"""Speaker-balanced batches for genuine-speaker contrastive learning."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler

from src.cached_dataset import CachedFeatureDataset


def build_speaker_index(
    dataset: CachedFeatureDataset,
) -> tuple[dict[str, list[int]], list[int]]:
    genuine_by_speaker: dict[str, list[int]] = defaultdict(list)
    spoof_indices: list[int] = []

    for index, record in enumerate(dataset.records):
        if record["label"] == 1:
            spoof_indices.append(index)
            continue
        with np.load(record["path"], allow_pickle=False) as data:
            metadata = (
                json.loads(str(data["metadata_json"]))
                if "metadata_json" in data
                else {}
            )
        speaker = str(metadata.get("speaker_id", "")).strip()
        if speaker:
            genuine_by_speaker[speaker].append(index)

    genuine_by_speaker = {
        speaker: indices
        for speaker, indices in genuine_by_speaker.items()
        if len(indices) >= 2
    }
    if not genuine_by_speaker:
        raise RuntimeError("No bona-fide speakers with at least two recordings found")
    if not spoof_indices:
        raise RuntimeError("No spoof recordings found")
    return genuine_by_speaker, spoof_indices


class SpeakerContrastiveBatchSampler(Sampler[list[int]]):
    """
    Each batch contains genuine pairs from multiple speakers plus spoof samples.

    Example for batch_size=32:
      8 speakers × 2 genuine recordings = 16 genuine
      16 spoof recordings
    """

    def __init__(
        self,
        dataset: CachedFeatureDataset,
        batch_size: int,
        speakers_per_batch: int | None = None,
        seed: int = 42,
    ) -> None:
        if batch_size < 4 or batch_size % 2:
            raise ValueError("batch_size must be an even integer of at least 4")
        self.dataset = dataset
        self.batch_size = batch_size
        self.speakers_per_batch = speakers_per_batch or batch_size // 4
        self.genuine_per_batch = self.speakers_per_batch * 2
        self.spoof_per_batch = batch_size - self.genuine_per_batch
        if self.spoof_per_batch < 1:
            raise ValueError("batch configuration leaves no room for spoof samples")
        self.genuine_by_speaker, self.spoof_indices = build_speaker_index(dataset)
        self.speakers = sorted(self.genuine_by_speaker)
        self.seed = seed
        self.epoch = 0
        self.batch_count = math.ceil(len(dataset) / batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batch_count

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.batch_count):
            selected_speakers = rng.choices(
                self.speakers, k=self.speakers_per_batch
            )
            batch: list[int] = []
            for speaker in selected_speakers:
                batch.extend(
                    rng.sample(self.genuine_by_speaker[speaker], k=2)
                )
            batch.extend(
                rng.choices(self.spoof_indices, k=self.spoof_per_batch)
            )
            rng.shuffle(batch)
            yield batch
