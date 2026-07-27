"""Forward/backward smoke test for Phase 2."""

from __future__ import annotations

import torch

from src.model_phase2 import Phase2GeneralDetector


def main() -> None:
    model = Phase2GeneralDetector(
        mfcc_channels=120,
        lfcc_channels=40,
        handcrafted_dim=74,
        xlsr_dim=1024,
    )
    batch = 4
    output = model(
        torch.randn(batch, 120, 201),
        torch.randn(batch, 40, 201),
        torch.randn(batch, 74),
        torch.randn(batch, 1024),
    )
    loss = torch.nn.functional.cross_entropy(
        output["logits"], torch.tensor([0, 1, 0, 1])
    )
    loss.backward()
    assert output["logits"].shape == (batch, 2)
    assert output["embedding"].shape == (batch, 256)
    assert output["branch_weights"].shape == (batch, 4)
    print("Phase 2 smoke test passed.")
    print("logits:", tuple(output["logits"].shape))
    print("embedding:", tuple(output["embedding"].shape))
    print("branch weights:", tuple(output["branch_weights"].shape))


if __name__ == "__main__":
    main()
