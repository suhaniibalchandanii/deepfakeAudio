"""Hybrid general detector using MFCC, LFCC, handcrafted, and XLS-R branches."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as functional


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.network = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.network(values)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(output_channels),
            nn.GELU(),
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(output_channels),
            SEBlock(output_channels),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Sequential(
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(output_channels),
            )
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return functional.gelu(self.main(values) + self.skip(values))


class SequenceEncoder(nn.Module):
    def __init__(self, channels: int, output_dim: int = 192) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(channels, 64, 5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            ResidualBlock(64, 96, stride=2),
            ResidualBlock(96, 128, stride=2),
            ResidualBlock(128, 192, stride=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(192, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(values))


class VectorEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        hidden = max(output_dim * 2, 128)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class GatedFusion(nn.Module):
    def __init__(self, branch_dim: int, branch_count: int = 4) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(branch_dim, branch_dim // 2),
            nn.Tanh(),
            nn.Linear(branch_dim // 2, 1),
        )
        self.branch_count = branch_count

    def forward(
        self, branches: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(branches, dim=1)
        weights = torch.softmax(self.scorer(stacked).squeeze(-1), dim=1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        return fused, weights


class Phase2GeneralDetector(nn.Module):
    def __init__(
        self,
        mfcc_channels: int,
        lfcc_channels: int,
        handcrafted_dim: int,
        xlsr_dim: int,
        branch_dim: int = 192,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.dimensions = {
            "mfcc_channels": mfcc_channels,
            "lfcc_channels": lfcc_channels,
            "handcrafted_dim": handcrafted_dim,
            "xlsr_dim": xlsr_dim,
            "branch_dim": branch_dim,
            "embedding_dim": embedding_dim,
        }
        self.mfcc_encoder = SequenceEncoder(mfcc_channels, branch_dim)
        self.lfcc_encoder = SequenceEncoder(lfcc_channels, branch_dim)
        self.handcrafted_encoder = VectorEncoder(handcrafted_dim, branch_dim)
        self.xlsr_encoder = VectorEncoder(xlsr_dim, branch_dim)
        self.fusion = GatedFusion(branch_dim, branch_count=4)
        self.projection = nn.Sequential(
            nn.Linear(branch_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Linear(embedding_dim, 2)

    def forward(
        self,
        mfcc: torch.Tensor,
        lfcc: torch.Tensor,
        handcrafted: torch.Tensor,
        xlsr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        branches = [
            self.mfcc_encoder(mfcc),
            self.lfcc_encoder(lfcc),
            self.handcrafted_encoder(handcrafted),
            self.xlsr_encoder(xlsr),
        ]
        fused, branch_weights = self.fusion(branches)
        embedding = functional.normalize(self.projection(fused), dim=-1)
        logits = self.classifier(embedding)
        return {
            "logits": logits,
            "embedding": embedding,
            "branch_weights": branch_weights,
        }
