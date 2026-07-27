"""Frozen XLS-R utterance embeddings with local Hugging Face caching."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoFeatureExtractor, Wav2Vec2Model

from src.config import Settings


class XLSREmbedder:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.device = torch.device(settings.device)
        self.processor = AutoFeatureExtractor.from_pretrained(settings.xlsr_model_name)
        self.model = Wav2Vec2Model.from_pretrained(settings.xlsr_model_name).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.inference_mode()
    def extract(self, waveform: np.ndarray) -> np.ndarray:
        inputs = self.processor(
            waveform,
            sampling_rate=self.s.sample_rate,
            return_tensors="pt",
        )
        outputs = self.model(
            input_values=inputs.input_values.to(self.device),
            attention_mask=(
                inputs.attention_mask.to(self.device)
                if "attention_mask" in inputs
                else None
            ),
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[self.s.xlsr_layer]
        embedding = hidden.mean(dim=1)
        embedding = functional.normalize(embedding, dim=-1)
        return embedding.squeeze(0).cpu().numpy().astype(np.float32)
