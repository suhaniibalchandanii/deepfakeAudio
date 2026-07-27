"""Forward and joint-loss smoke test for Phase 3."""

import torch

from src.losses_phase3 import Phase3JointLoss
from src.model_phase3 import Phase3SpeakerAwareDetector


model = Phase3SpeakerAwareDetector(120, 40, 74, 1024)
output = model(
    torch.randn(8, 120, 201),
    torch.randn(8, 40, 201),
    torch.randn(8, 74),
    torch.randn(8, 1024),
)
labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
speakers = ["A", "A", "B", "B", "A", "B", "C", "D"]
losses = Phase3JointLoss()(output, labels, speakers)
losses["total"].backward()
assert output["speaker_embedding"].shape == (8, 128)
assert torch.isfinite(losses["total"])
print("Phase 3 smoke test passed.")
print({key: float(value) for key, value in losses.items()})
