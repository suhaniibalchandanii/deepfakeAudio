"""Phase 1 hybrid feature bundle creation."""

from __future__ import annotations

import numpy as np


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    mean = sequence.mean(axis=1, keepdims=True)
    std = np.maximum(sequence.std(axis=1, keepdims=True), 1e-6)
    return ((sequence - mean) / std).astype(np.float32)


def combine_features(
    handcrafted: dict[str, np.ndarray],
    xlsr_embedding: np.ndarray | None,
) -> dict[str, np.ndarray]:
    global_vector = handcrafted["handcrafted_global"]
    if xlsr_embedding is None:
        xlsr_embedding = np.empty((0,), dtype=np.float32)
    hybrid_vector = np.concatenate([global_vector, xlsr_embedding]).astype(np.float32)
    return {
        "mfcc": normalize_sequence(handcrafted["mfcc"]),
        "lfcc": normalize_sequence(handcrafted["lfcc"]),
        "spectral": handcrafted["spectral"],
        "prosodic": handcrafted["prosodic"],
        "physics": handcrafted["physics"],
        "handcrafted_global": global_vector,
        "xlsr": xlsr_embedding,
        "hybrid_vector": hybrid_vector,
    }
