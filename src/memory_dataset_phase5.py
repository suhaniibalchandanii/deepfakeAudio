"""Episodic candidate-reference dataset for Phase 5."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import PROJECT_ROOT


class MemoryEpisodeDataset(Dataset):
    """Pairs each query with bona-fide candidate references of its speaker."""

    def __init__(
        self,
        split: str,
        references_per_query: int = 4,
        seed: int = 42,
    ) -> None:
        path = PROJECT_ROOT / "cache" / "phase5" / f"{split}_embeddings.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Phase 5 cache not found: {path}\n"
                f"Run: python -m src.cache_phase5_embeddings --split {split}"
            )
        with np.load(path, allow_pickle=False) as data:
            self.audio_ids = data["audio_ids"].astype(str)
            self.speaker_ids = data["speaker_ids"].astype(str)
            self.labels = data["labels"].astype(np.int64)
            self.general_logits = data["general_logits"].astype(np.float32)
            self.general_embeddings = data["general_embeddings"].astype(np.float32)
            self.speaker_embeddings = data["speaker_embeddings"].astype(np.float32)

        self.split = split
        self.references_per_query = references_per_query
        self.seed = seed
        self.epoch = 0
        bona_fide_by_speaker: dict[str, list[int]] = defaultdict(list)
        for index, (speaker_id, label) in enumerate(
            zip(self.speaker_ids, self.labels)
        ):
            if label == 0 and speaker_id:
                bona_fide_by_speaker[str(speaker_id)].append(index)
        self.bona_fide_by_speaker = dict(bona_fide_by_speaker)

        self.query_indices = []
        for index, speaker_id in enumerate(self.speaker_ids):
            candidates = self.bona_fide_by_speaker.get(str(speaker_id), [])
            candidates = [candidate for candidate in candidates if candidate != index]
            if candidates:
                self.query_indices.append(index)
        if not self.query_indices:
            raise RuntimeError("No queries have a separate bona-fide reference.")
        print(
            f"{split}: {len(self.query_indices)} queries with candidate references "
            f"from {len(self.bona_fide_by_speaker)} speakers"
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.query_indices)

    def _reference_indices(self, query_index: int) -> np.ndarray:
        speaker_id = str(self.speaker_ids[query_index])
        candidates = np.asarray(
            [
                index
                for index in self.bona_fide_by_speaker[speaker_id]
                if index != query_index
            ],
            dtype=np.int64,
        )
        rng = np.random.default_rng(
            self.seed + self.epoch * 1_000_003 + int(query_index)
        )
        replace = len(candidates) < self.references_per_query
        return rng.choice(
            candidates,
            size=self.references_per_query,
            replace=replace,
        )

    def __getitem__(self, item: int) -> dict[str, object]:
        query_index = self.query_indices[item]
        reference_indices = self._reference_indices(query_index)
        return {
            "audio_id": str(self.audio_ids[query_index]),
            "speaker_id": str(self.speaker_ids[query_index]),
            "label": torch.tensor(self.labels[query_index], dtype=torch.long),
            "general_logits": torch.from_numpy(
                self.general_logits[query_index]
            ),
            "general_embedding": torch.from_numpy(
                self.general_embeddings[query_index]
            ),
            "speaker_embedding": torch.from_numpy(
                self.speaker_embeddings[query_index]
            ),
            "reference_embeddings": torch.from_numpy(
                self.speaker_embeddings[reference_indices]
            ),
            "reference_audio_ids": [
                str(self.audio_ids[index]) for index in reference_indices
            ],
        }
