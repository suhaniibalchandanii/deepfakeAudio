"""Classification plus genuine-speaker supervised contrastive loss."""

from __future__ import annotations

import torch
from torch import nn


class GenuineSpeakerContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss over bona-fide samples only.

    Spoof samples are deliberately excluded from the speaker objective because
    their speaker identity is not a trusted enrollment target.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        speaker_ids: list[str],
    ) -> torch.Tensor:
        genuine_indices = torch.nonzero(labels == 0, as_tuple=False).squeeze(1)
        if genuine_indices.numel() < 2:
            return embeddings.sum() * 0.0

        genuine_embeddings = embeddings[genuine_indices]
        genuine_speakers = [
            speaker_ids[int(index)] for index in genuine_indices.cpu()
        ]
        similarities = (
            genuine_embeddings @ genuine_embeddings.T
        ) / self.temperature
        count = similarities.size(0)
        eye = torch.eye(count, device=similarities.device, dtype=torch.bool)
        positive_mask = torch.zeros_like(eye)
        for row in range(count):
            for column in range(count):
                positive_mask[row, column] = (
                    row != column
                    and genuine_speakers[row]
                    and genuine_speakers[row] == genuine_speakers[column]
                )

        valid_anchor = positive_mask.any(dim=1)
        if not valid_anchor.any():
            return embeddings.sum() * 0.0

        masked_logits = similarities.masked_fill(eye, float("-inf"))
        log_denominator = torch.logsumexp(masked_logits, dim=1)
        positive_logits = similarities.masked_fill(
            ~positive_mask, 0.0
        ).sum(dim=1)
        positive_count = positive_mask.sum(dim=1).clamp_min(1)
        mean_positive = positive_logits / positive_count
        return (log_denominator[valid_anchor] - mean_positive[valid_anchor]).mean()


class Phase3JointLoss(nn.Module):
    def __init__(
        self,
        contrastive_weight: float = 0.15,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.classification = nn.CrossEntropyLoss()
        self.contrastive = GenuineSpeakerContrastiveLoss(temperature)
        self.contrastive_weight = contrastive_weight

    def forward(
        self,
        output: dict[str, torch.Tensor],
        labels: torch.Tensor,
        speaker_ids: list[str],
    ) -> dict[str, torch.Tensor]:
        classification_loss = self.classification(output["logits"], labels)
        contrastive_loss = self.contrastive(
            output["speaker_embedding"], labels, speaker_ids
        )
        total = (
            classification_loss
            + self.contrastive_weight * contrastive_loss
        )
        return {
            "total": total,
            "classification": classification_loss,
            "contrastive": contrastive_loss,
        }
