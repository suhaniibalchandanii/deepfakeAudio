"""ASVspoof2021 LA protocol parsing and Phase 8 cached datasets."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PROTOCOL_COLUMNS = [
    "speaker_id",
    "audio_id",
    "codec",
    "transmission",
    "attack_id",
    "label_text",
    "trim",
    "subset",
]
LABEL_MAP = {"bonafide": 0, "spoof": 1}


def parse_asvspoof2021_la_protocol(path: Path) -> pd.DataFrame:
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 8:
                raise ValueError(
                    f"Expected at least 8 protocol columns at line "
                    f"{line_number}, found {len(parts)}."
                )
            values = dict(zip(PROTOCOL_COLUMNS, parts[:8]))
            label = values["label_text"].lower()
            if label not in LABEL_MAP:
                raise ValueError(
                    f"Unknown label {label!r} at line {line_number}."
                )
            values["label"] = LABEL_MAP[label]
            rows.append(values)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No protocol rows found in {path}")
    return frame


def select_reproducible_subset(
    frame: pd.DataFrame,
    max_samples: int | None,
    balanced: bool,
    seed: int,
) -> pd.DataFrame:
    if max_samples is None or max_samples >= len(frame):
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")
    if not balanced:
        return frame.sample(n=max_samples, random_state=seed).reset_index(
            drop=True
        )
    counts = {0: max_samples // 2, 1: max_samples - max_samples // 2}
    selected = []
    for label, requested in counts.items():
        group = frame[frame["label"] == label]
        if len(group) < requested:
            raise ValueError(
                f"Requested {requested} label={label} rows, only "
                f"{len(group)} are available."
            )
        selected.append(
            group.sample(n=requested, random_state=seed + label)
        )
    return (
        pd.concat(selected)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def build_audio_index(audio_root: Path) -> dict[str, Path]:
    extensions = ("*.flac", "*.wav")
    paths = []
    for extension in extensions:
        paths.extend(Path(audio_root).rglob(extension))
    index = {}
    duplicates = set()
    for path in paths:
        if path.stem in index:
            duplicates.add(path.stem)
        else:
            index[path.stem] = path
    if duplicates:
        print(
            f"Warning: {len(duplicates)} duplicate audio stems were found; "
            "the first path is used."
        )
    if not index:
        raise FileNotFoundError(
            f"No FLAC or WAV audio found below {audio_root}"
        )
    return index


class Phase8CachedFeatureDataset(Dataset):
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        frame = pd.read_csv(manifest_path)
        records = []
        for row in frame.itertuples(index=False):
            path = self.cache_dir / f"{row.audio_id}.npz"
            if path.exists():
                records.append(
                    {
                        "path": path,
                        "audio_id": str(row.audio_id),
                        "speaker_id": str(row.speaker_id),
                        "label": int(row.label),
                    }
                )
        if not records:
            raise RuntimeError(f"No usable caches in {self.cache_dir}")
        self.records = records
        print(f"Phase 8 cached samples: {len(records)}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with np.load(record["path"], allow_pickle=False) as data:
            return {
                "mfcc": torch.from_numpy(data["mfcc"].astype(np.float32)),
                "lfcc": torch.from_numpy(data["lfcc"].astype(np.float32)),
                "handcrafted": torch.from_numpy(
                    data["handcrafted_global"].astype(np.float32)
                ),
                "xlsr": torch.from_numpy(data["xlsr"].astype(np.float32)),
                "label": torch.tensor(record["label"], dtype=torch.long),
                "audio_id": record["audio_id"],
                "speaker_id": record["speaker_id"],
            }


class Phase8MemoryDataset(Dataset):
    """Creates 2021 query/reference episodes from frozen embeddings."""

    def __init__(
        self,
        embedding_path: Path,
        references_per_query: int,
        seed: int = 42,
    ) -> None:
        with np.load(embedding_path, allow_pickle=False) as data:
            self.audio_ids = data["audio_ids"].astype(str)
            self.speaker_ids = data["speaker_ids"].astype(str)
            self.labels = data["labels"].astype(np.int64)
            self.general_logits = data["general_logits"].astype(np.float32)
            self.general_embeddings = data["general_embeddings"].astype(
                np.float32
            )
            self.speaker_embeddings = data["speaker_embeddings"].astype(
                np.float32
            )
        self.references_per_query = references_per_query
        self.seed = seed
        bona_fide: dict[str, list[int]] = defaultdict(list)
        for index, (speaker, label) in enumerate(
            zip(self.speaker_ids, self.labels)
        ):
            if label == 0 and speaker:
                bona_fide[str(speaker)].append(index)
        self.bona_fide = dict(bona_fide)
        self.query_indices = []
        for index, speaker in enumerate(self.speaker_ids):
            references = [
                candidate
                for candidate in self.bona_fide.get(str(speaker), [])
                if candidate != index
            ]
            if references:
                self.query_indices.append(index)
        if not self.query_indices:
            raise RuntimeError(
                "No queries have a separate same-speaker bona-fide reference."
            )
        print(
            f"Phase 8: {len(self.query_indices)} queries, "
            f"{len(self.bona_fide)} speakers, "
            f"{references_per_query} references/query"
        )

    def __len__(self) -> int:
        return len(self.query_indices)

    def __getitem__(self, item: int) -> dict[str, object]:
        query = self.query_indices[item]
        candidates = np.asarray(
            [
                index
                for index in self.bona_fide[str(self.speaker_ids[query])]
                if index != query
            ],
            dtype=np.int64,
        )
        rng = np.random.default_rng(self.seed + int(query))
        references = rng.choice(
            candidates,
            size=self.references_per_query,
            replace=len(candidates) < self.references_per_query,
        )
        return {
            "audio_id": str(self.audio_ids[query]),
            "speaker_id": str(self.speaker_ids[query]),
            "label": torch.tensor(self.labels[query], dtype=torch.long),
            "general_logits": torch.from_numpy(self.general_logits[query]),
            "general_embedding": torch.from_numpy(
                self.general_embeddings[query]
            ),
            "speaker_embedding": torch.from_numpy(
                self.speaker_embeddings[query]
            ),
            "reference_embeddings": torch.from_numpy(
                self.speaker_embeddings[references]
            ),
        }
