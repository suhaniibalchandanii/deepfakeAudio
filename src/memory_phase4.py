"""Persistent per-user speaker memory with exact cosine retrieval."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot store or search a zero embedding")
    return vector / norm


def safe_user_key(user_id: str) -> str:
    if not user_id.strip():
        raise ValueError("user_id cannot be empty")
    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:20]
    return f"user_{digest}"


@dataclass
class SearchResult:
    embeddings: np.ndarray
    similarities: np.ndarray
    audio_ids: list[str]
    timestamps: list[str]


class UserMemory:
    def __init__(
        self,
        root: Path,
        embedding_dim: int = 128,
        max_entries: int = 20,
        diversity_similarity: float = 0.98,
    ) -> None:
        self.root = Path(root)
        self.embedding_dim = embedding_dim
        self.max_entries = max_entries
        self.diversity_similarity = diversity_similarity
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        return self.root / f"{safe_user_key(user_id)}.npz"

    def exists(self, user_id: str) -> bool:
        return self.path_for(user_id).is_file()

    def load(self, user_id: str) -> tuple[np.ndarray, list[str], list[str]]:
        path = self.path_for(user_id)
        if not path.is_file():
            return (
                np.empty((0, self.embedding_dim), dtype=np.float32),
                [],
                [],
            )
        with np.load(path, allow_pickle=False) as data:
            embeddings = data["embeddings"].astype(np.float32)
            audio_ids = data["audio_ids"].astype(str).tolist()
            timestamps = data["timestamps"].astype(str).tolist()
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Invalid memory shape in {path}: {embeddings.shape}")
        return embeddings, audio_ids, timestamps

    def search(
        self,
        user_id: str,
        query: np.ndarray,
        top_k: int = 5,
    ) -> SearchResult:
        embeddings, audio_ids, timestamps = self.load(user_id)
        if len(embeddings) == 0:
            return SearchResult(
                embeddings=embeddings,
                similarities=np.empty((0,), dtype=np.float32),
                audio_ids=[],
                timestamps=[],
            )
        query = normalize(query)
        similarities = embeddings @ query
        indices = np.argsort(similarities)[::-1][: min(top_k, len(embeddings))]
        return SearchResult(
            embeddings=embeddings[indices],
            similarities=similarities[indices].astype(np.float32),
            audio_ids=[audio_ids[index] for index in indices],
            timestamps=[timestamps[index] for index in indices],
        )

    def add(
        self,
        user_id: str,
        embedding: np.ndarray,
        audio_id: str,
        force: bool = False,
    ) -> dict[str, object]:
        embedding = normalize(embedding)
        if embedding.shape != (self.embedding_dim,):
            raise ValueError(
                f"Expected embedding shape {(self.embedding_dim,)}, "
                f"received {embedding.shape}"
            )
        embeddings, audio_ids, timestamps = self.load(user_id)

        if len(embeddings) and not force:
            maximum_similarity = float(np.max(embeddings @ embedding))
            if maximum_similarity >= self.diversity_similarity:
                return {
                    "added": False,
                    "reason": "near_duplicate",
                    "maximum_similarity": maximum_similarity,
                    "memory_size": len(embeddings),
                }

        timestamp = datetime.now(timezone.utc).isoformat()
        embeddings = np.vstack([embeddings, embedding[None, :]])
        audio_ids.append(str(audio_id))
        timestamps.append(timestamp)

        # Keep a bounded, recent memory. A later phase can replace this with
        # diversity-aware prototype selection.
        if len(embeddings) > self.max_entries:
            embeddings = embeddings[-self.max_entries :]
            audio_ids = audio_ids[-self.max_entries :]
            timestamps = timestamps[-self.max_entries :]

        path = self.path_for(user_id)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            embeddings=embeddings.astype(np.float32),
            audio_ids=np.asarray(audio_ids, dtype=str),
            timestamps=np.asarray(timestamps, dtype=str),
        )
        os.replace(temporary, path)
        return {
            "added": True,
            "reason": "enrolled" if len(embeddings) == 1 else "updated",
            "memory_size": len(embeddings),
            "path": str(path),
        }
