"""Shape and gradient smoke test for learned memory attention."""

import torch

from src.model_phase5 import Phase5MemoryAttention


model = Phase5MemoryAttention()
batch_size, references = 3, 4
general_logits = torch.randn(batch_size, 2)
general_embedding = torch.randn(batch_size, 256)
speaker_embedding = torch.randn(batch_size, 128)
reference_embeddings = torch.randn(batch_size, references, 128)
output = model(
    general_logits,
    general_embedding,
    speaker_embedding,
    reference_embeddings,
)
assert output["logits"].shape == (batch_size, 2)
assert output["attention_weights"].shape == (batch_size, references)
assert torch.allclose(output["logits"], general_logits, atol=1e-6)
torch.nn.functional.cross_entropy(
    output["logits"], torch.tensor([0, 1, 0])
).backward()
assert any(parameter.grad is not None for parameter in model.parameters())
print("Phase 5 learned memory-attention smoke test passed.")
