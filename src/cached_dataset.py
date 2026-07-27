"""Dataset for Phase 1 NPZ caches; no audio or XLS-R computation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import SETTINGS


class CachedFeatureDataset(Dataset):
    def __init__(
        self,
        split: str,
        cache_root: Path | None = None,
        require_xlsr: bool = True,
    ) -> None:
        self.split = split
        self.cache_dir = (cache_root or SETTINGS.cache_root) / split
        self.manifest_path = self.cache_dir / "manifest.csv"
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        manifest = pd.read_csv(self.manifest_path)
        required = {"audio_id", "label"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        records = []
        skipped = []
        for row in manifest.itertuples(index=False):
            path = self.cache_dir / f"{row.audio_id}.npz"
            if not path.is_file():
                skipped.append((row.audio_id, "missing cache"))
                continue
            if require_xlsr:
                try:
                    with np.load(path, allow_pickle=False) as data:
                        if "xlsr" not in data or data["xlsr"].size == 0:
                            skipped.append((row.audio_id, "missing XLS-R"))
                            continue
                except Exception as error:
                    skipped.append((row.audio_id, str(error)))
                    continue
            records.append(
                {
                    "audio_id": str(row.audio_id),
                    "label": int(row.label),
                    "path": path,
                }
            )

        if not records:
            raise RuntimeError(f"No usable cached samples found in {self.cache_dir}")
        self.records = records
        self.skipped = skipped
        labels = np.asarray([record["label"] for record in records])
        self.class_counts = {
            int(label): int(count)
            for label, count in zip(*np.unique(labels, return_counts=True))
        }
        print(
            f"{split}: {len(records)} usable caches, "
            f"{len(skipped)} skipped, class counts={self.class_counts}"
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with np.load(record["path"], allow_pickle=False) as data:
            metadata = (
                json.loads(str(data["metadata_json"]))
                if "metadata_json" in data
                else {}
            )
            return {
                "mfcc": torch.from_numpy(data["mfcc"].astype(np.float32)),
                "lfcc": torch.from_numpy(data["lfcc"].astype(np.float32)),
                "handcrafted": torch.from_numpy(
                    data["handcrafted_global"].astype(np.float32)
                ),
                "xlsr": torch.from_numpy(data["xlsr"].astype(np.float32)),
                "label": torch.tensor(record["label"], dtype=torch.long),
                "audio_id": record["audio_id"],
                "speaker_id": str(metadata.get("speaker_id", "")),
                "attack_id": str(metadata.get("attack_id", "")),
                "candidate_reference": bool(
                    metadata.get("candidate_reference", False)
                ),
            }


def infer_feature_dimensions(dataset: CachedFeatureDataset) -> dict[str, int]:
    sample = dataset[0]
    return {
        "mfcc_channels": int(sample["mfcc"].shape[0]),
        "lfcc_channels": int(sample["lfcc"].shape[0]),
        "handcrafted_dim": int(sample["handcrafted"].numel()),
        "xlsr_dim": int(sample["xlsr"].numel()),
    }
