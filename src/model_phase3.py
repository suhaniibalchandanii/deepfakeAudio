"""Phase 3 joint spoof-classification and speaker-contrastive model."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as functional

from src.model_phase2 import Phase2GeneralDetector


class Phase3SpeakerAwareDetector(Phase2GeneralDetector):
    def __init__(
        self,
        mfcc_channels: int,
        lfcc_channels: int,
        handcrafted_dim: int,
        xlsr_dim: int,
        branch_dim: int = 192,
        embedding_dim: int = 256,
        speaker_embedding_dim: int = 128,
    ) -> None:
        super().__init__(
            mfcc_channels=mfcc_channels,
            lfcc_channels=lfcc_channels,
            handcrafted_dim=handcrafted_dim,
            xlsr_dim=xlsr_dim,
            branch_dim=branch_dim,
            embedding_dim=embedding_dim,
        )
        self.dimensions["speaker_embedding_dim"] = speaker_embedding_dim
        self.speaker_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(embedding_dim, speaker_embedding_dim),
        )

    def forward(
        self,
        mfcc: torch.Tensor,
        lfcc: torch.Tensor,
        handcrafted: torch.Tensor,
        xlsr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = super().forward(mfcc, lfcc, handcrafted, xlsr)
        output["speaker_embedding"] = functional.normalize(
            self.speaker_projection(output["embedding"]), dim=-1
        )
        return output


def load_phase2_weights(
    model: Phase3SpeakerAwareDetector,
    checkpoint: dict,
) -> tuple[list[str], list[str]]:
    result = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    return list(result.missing_keys), list(result.unexpected_keys)
