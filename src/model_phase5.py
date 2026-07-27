"""Learned cross-attention fusion for general and personalized evidence."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as functional


class Phase5MemoryAttention(nn.Module):
    def __init__(
        self,
        general_embedding_dim: int = 256,
        speaker_embedding_dim: int = 128,
        attention_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.dimensions = {
            "general_embedding_dim": general_embedding_dim,
            "speaker_embedding_dim": speaker_embedding_dim,
            "attention_dim": attention_dim,
            "heads": heads,
        }
        self.query_projection = nn.Linear(
            general_embedding_dim + speaker_embedding_dim, attention_dim
        )
        self.memory_projection = nn.Linear(speaker_embedding_dim, attention_dim)
        self.cross_attention = nn.MultiheadAttention(
            attention_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        fusion_dim = (
            general_embedding_dim
            + speaker_embedding_dim
            + attention_dim
            + 2  # mean and maximum speaker cosine similarity
            + 2  # frozen general detector probabilities
        )
        self.normalization = nn.LayerNorm(fusion_dim)
        self.correction = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )
        self.gate = nn.Sequential(nn.Linear(fusion_dim, 1), nn.Sigmoid())

        # Training starts exactly at the already-strong general detector.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        nn.init.zeros_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, -1.0)

    def forward(
        self,
        general_logits: torch.Tensor,
        general_embedding: torch.Tensor,
        speaker_embedding: torch.Tensor,
        reference_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        speaker_embedding = functional.normalize(speaker_embedding, dim=-1)
        reference_embeddings = functional.normalize(
            reference_embeddings, dim=-1
        )
        query = self.query_projection(
            torch.cat([general_embedding, speaker_embedding], dim=-1)
        ).unsqueeze(1)
        memory = self.memory_projection(reference_embeddings)
        attended, attention_weights = self.cross_attention(
            query=query,
            key=memory,
            value=memory,
            need_weights=True,
        )
        similarities = torch.einsum(
            "bd,bkd->bk", speaker_embedding, reference_embeddings
        )
        general_probabilities = torch.softmax(general_logits, dim=-1)
        fusion = torch.cat(
            [
                general_embedding,
                speaker_embedding,
                attended.squeeze(1),
                similarities.mean(dim=1, keepdim=True),
                similarities.max(dim=1, keepdim=True).values,
                general_probabilities,
            ],
            dim=-1,
        )
        fusion = self.normalization(fusion)
        gate = self.gate(fusion)
        correction = self.correction(fusion)
        personalized_logits = general_logits + gate * correction
        return {
            "logits": personalized_logits,
            "general_logits": general_logits,
            "correction": correction,
            "gate": gate.squeeze(-1),
            "attention_weights": attention_weights.squeeze(1),
            "mean_similarity": similarities.mean(dim=1),
            "max_similarity": similarities.max(dim=1).values,
        }
