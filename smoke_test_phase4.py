"""Safe enrollment and memory-update smoke test."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from src.memory_phase4 import UserMemory


with tempfile.TemporaryDirectory() as temporary:
    memory = UserMemory(Path(temporary), embedding_dim=128)
    reference = np.zeros(128, dtype=np.float32)
    reference[0] = 1.0
    similar = reference.copy()
    similar[1] = 0.05
    different = np.zeros(128, dtype=np.float32)
    different[2] = 1.0

    added = memory.add("test-user", reference, "first-input")
    assert added["added"] and memory.exists("test-user")
    result = memory.search("test-user", similar, top_k=1)
    assert result.similarities[0] > 0.99
    result = memory.search("test-user", different, top_k=1)
    assert result.similarities[0] < 0.01
    duplicate = memory.add("test-user", similar, "near-duplicate")
    assert not duplicate["added"]
    print("Phase 4 memory smoke test passed.")