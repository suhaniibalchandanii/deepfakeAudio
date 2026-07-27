"""General detection plus confidence-gated candidate-reference memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.config import PROJECT_ROOT, SETTINGS
from src.memory_phase4 import UserMemory
from src.model_phase2 import Phase2GeneralDetector
from src.model_phase3 import Phase3SpeakerAwareDetector


@dataclass(frozen=True)
class DecisionThresholds:
    initial_enrollment_bonafide: float = 0.98
    general_bonafide_required: float = 0.90
    speaker_similarity: float = 0.40
    similarity_temperature: float = 0.08
    general_weight: float = 0.70
    speaker_weight: float = 0.30
    final_bonafide: float = 0.50
    update_final_bonafide: float = 0.95
    update_general_bonafide: float = 0.98


class PersonalizedDetector:
    def __init__(
        self,
        memory_root: Path | None = None,
        thresholds: DecisionThresholds = DecisionThresholds(),
    ) -> None:
        self.device = torch.device(SETTINGS.device)
        self.thresholds = thresholds
        self.general_model = self._load_general_model()
        self.speaker_model = self._load_speaker_model()
        speaker_dim = int(
            self.speaker_model.dimensions["speaker_embedding_dim"]
        )
        self.memory = UserMemory(
            root=memory_root or (PROJECT_ROOT / "memory" / "users"),
            embedding_dim=speaker_dim,
        )

    def _load_general_model(self) -> Phase2GeneralDetector:
        path = PROJECT_ROOT / "checkpoints" / "phase2_best.pth"
        checkpoint = torch.load(path, map_location=self.device)
        model = Phase2GeneralDetector(**checkpoint["dimensions"]).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def _load_speaker_model(self) -> Phase3SpeakerAwareDetector:
        path = PROJECT_ROOT / "checkpoints" / "phase3_best.pth"
        checkpoint = torch.load(path, map_location=self.device)
        model = Phase3SpeakerAwareDetector(**checkpoint["dimensions"]).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    @staticmethod
    def _batch_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
        return value.unsqueeze(0).to(device)

    @torch.inference_mode()
    def encode_cached_sample(
        self,
        sample: dict[str, object],
    ) -> dict[str, object]:
        arguments = (
            self._batch_tensor(sample["mfcc"], self.device),
            self._batch_tensor(sample["lfcc"], self.device),
            self._batch_tensor(sample["handcrafted"], self.device),
            self._batch_tensor(sample["xlsr"], self.device),
        )
        general_output = self.general_model(*arguments)
        speaker_output = self.speaker_model(*arguments)
        probability = torch.softmax(general_output["logits"], dim=1)[0]
        return {
            "general_bonafide_probability": float(probability[0].cpu()),
            "general_spoof_probability": float(probability[1].cpu()),
            "general_prediction": int(torch.argmax(probability).cpu()),
            "speaker_embedding": (
                speaker_output["speaker_embedding"][0].cpu().numpy()
            ),
            "hybrid_embedding": general_output["embedding"][0].cpu().numpy(),
        }

    def similarity_probability(self, similarity: float) -> float:
        scaled = (
            similarity - self.thresholds.speaker_similarity
        ) / self.thresholds.similarity_temperature
        return float(1.0 / (1.0 + np.exp(-np.clip(scaled, -30, 30))))

    def process_cached_sample(
        self,
        user_id: str,
        sample: dict[str, object],
        allow_memory_update: bool = True,
    ) -> dict[str, object]:
        encoded = self.encode_cached_sample(sample)
        general_bonafide = encoded["general_bonafide_probability"]
        speaker_embedding = encoded["speaker_embedding"]
        audio_id = str(sample["audio_id"])

        result: dict[str, object] = {
            "user_id": user_id,
            "audio_id": audio_id,
            "thresholds": asdict(self.thresholds),
            **encoded,
            "memory_initialized": False,
            "memory_updated": False,
        }

        # No reference exists. The current input is only a candidate reference
        # until the general detector passes the strict enrollment threshold.
        if not self.memory.exists(user_id):
            accepted = (
                encoded["general_prediction"] == 0
                and general_bonafide
                >= self.thresholds.initial_enrollment_bonafide
            )
            result.update(
                {
                    "mode": "initial_candidate_reference",
                    "speaker_similarity": None,
                    "speaker_match_probability": None,
                    "personalized_bonafide_probability": general_bonafide,
                    "final_prediction": 0 if accepted else 1,
                    "decision": (
                        "bonafide_reference_enrolled"
                        if accepted
                        else "spoof_or_untrusted_not_enrolled"
                    ),
                }
            )
            if accepted and allow_memory_update:
                update = self.memory.add(user_id, speaker_embedding, audio_id)
                result["memory_initialized"] = bool(update["added"])
                result["memory_result"] = update
            return result

        neighbors = self.memory.search(user_id, speaker_embedding, top_k=5)
        similarity = (
            float(neighbors.similarities.mean())
            if len(neighbors.similarities)
            else -1.0
        )
        speaker_probability = self.similarity_probability(similarity)
        personalized_bonafide = (
            self.thresholds.general_weight * general_bonafide
            + self.thresholds.speaker_weight * speaker_probability
        )
        accepted = (
            general_bonafide >= self.thresholds.general_bonafide_required
            and similarity >= self.thresholds.speaker_similarity
            and personalized_bonafide >= self.thresholds.final_bonafide
        )
        result.update(
            {
                "mode": "personalized_verification",
                "speaker_similarity": similarity,
                "speaker_match_probability": speaker_probability,
                "personalized_bonafide_probability": personalized_bonafide,
                "final_prediction": 0 if accepted else 1,
                "decision": "bonafide" if accepted else "spoof_or_speaker_mismatch",
                "neighbor_audio_ids": neighbors.audio_ids,
                "neighbor_similarities": neighbors.similarities.tolist(),
            }
        )

        should_update = (
            accepted
            and allow_memory_update
            and general_bonafide
            >= self.thresholds.update_general_bonafide
            and personalized_bonafide
            >= self.thresholds.update_final_bonafide
            and similarity >= self.thresholds.speaker_similarity
        )
        if should_update:
            update = self.memory.add(user_id, speaker_embedding, audio_id)
            result["memory_updated"] = bool(update["added"])
            result["memory_result"] = update
        return result
