"""Production-style input-as-candidate-reference inference for Phase 6."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.config import PROJECT_ROOT, SETTINGS
from src.memory_phase4 import UserMemory, safe_user_key
from src.model_phase2 import Phase2GeneralDetector
from src.model_phase3 import Phase3SpeakerAwareDetector
from src.model_phase5 import Phase5MemoryAttention


class Phase6OnlineDetector:
    def __init__(self, memory_root: Path | None = None) -> None:
        self.device = torch.device(SETTINGS.device)
        self.calibration = self._load_calibration()
        self.general_model = self._load_phase2()
        self.speaker_model = self._load_phase3()
        self.attention_model, self.references = self._load_phase5()
        self.memory = UserMemory(
            memory_root or PROJECT_ROOT / "memory" / "users",
            embedding_dim=int(
                self.speaker_model.dimensions["speaker_embedding_dim"]
            ),
        )
        self.audit_path = PROJECT_ROOT / "outputs" / "phase6" / "audit.jsonl"
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_calibration(self) -> dict:
        path = PROJECT_ROOT / "checkpoints" / "phase6_calibration.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing calibration: {path}\n"
                "Run: python -m src.calibrate_phase6"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_phase2(self):
        checkpoint = torch.load(
            PROJECT_ROOT / "checkpoints" / "phase2_best.pth",
            map_location=self.device,
        )
        model = Phase2GeneralDetector(**checkpoint["dimensions"]).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def _load_phase3(self):
        checkpoint = torch.load(
            PROJECT_ROOT / "checkpoints" / "phase3_best.pth",
            map_location=self.device,
        )
        model = Phase3SpeakerAwareDetector(**checkpoint["dimensions"]).to(
            self.device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def _load_phase5(self):
        checkpoint = torch.load(
            PROJECT_ROOT / "checkpoints" / "phase5_best.pth",
            map_location=self.device,
        )
        model = Phase5MemoryAttention(**checkpoint["dimensions"]).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, int(checkpoint.get("references_per_query", 4))

    def _arguments(self, sample):
        return (
            sample["mfcc"].unsqueeze(0).to(self.device),
            sample["lfcc"].unsqueeze(0).to(self.device),
            sample["handcrafted"].unsqueeze(0).to(self.device),
            sample["xlsr"].unsqueeze(0).to(self.device),
        )

    @torch.inference_mode()
    def encode(self, sample: dict[str, object]) -> dict[str, object]:
        arguments = self._arguments(sample)
        general = self.general_model(*arguments)
        speaker = self.speaker_model(*arguments)
        probabilities = torch.softmax(general["logits"], dim=-1)[0]
        return {
            "general_logits": general["logits"],
            "general_embedding": general["embedding"],
            "speaker_embedding_tensor": speaker["speaker_embedding"],
            "speaker_embedding": (
                speaker["speaker_embedding"][0].cpu().numpy().astype(np.float32)
            ),
            "general_bonafide_probability": float(probabilities[0].cpu()),
            "general_spoof_probability": float(probabilities[1].cpu()),
        }

    def _padded_references(self, embeddings: np.ndarray) -> torch.Tensor:
        if len(embeddings) == 0:
            raise ValueError("At least one reference embedding is required.")
        if len(embeddings) < self.references:
            indices = np.resize(np.arange(len(embeddings)), self.references)
            embeddings = embeddings[indices]
        else:
            embeddings = embeddings[: self.references]
        return torch.from_numpy(embeddings).unsqueeze(0).to(self.device)

    def _audit(self, result: dict[str, object]) -> None:
        record = {
            key: value
            for key, value in result.items()
            if key not in {"neighbor_audio_ids"}
        }
        with self.audit_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

    @torch.inference_mode()
    def process(
        self,
        user_id: str,
        sample: dict[str, object],
        allow_memory_update: bool = True,
    ) -> dict[str, object]:
        encoded = self.encode(sample)
        audio_id = str(sample["audio_id"])
        general_bonafide = encoded["general_bonafide_probability"]
        general_spoof = encoded["general_spoof_probability"]
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_key": safe_user_key(user_id),
            "audio_id": audio_id,
            "general_bonafide_probability": general_bonafide,
            "general_spoof_probability": general_spoof,
            "memory_initialized": False,
            "memory_updated": False,
        }

        if not self.memory.exists(user_id):
            accepted = (
                general_bonafide
                >= self.calibration["initial_enrollment_bonafide_probability"]
            )
            result.update(
                {
                    "mode": "initial_candidate_reference",
                    "personalized_bonafide_probability": general_bonafide,
                    "personalized_spoof_probability": general_spoof,
                    "speaker_similarity": None,
                    "final_prediction": 0 if accepted else 1,
                    "decision": (
                        "bonafide_reference_enrolled"
                        if accepted and allow_memory_update
                        else (
                            "bonafide_candidate_not_saved"
                            if accepted
                            else "spoof_or_untrusted_not_enrolled"
                        )
                    ),
                }
            )
            if accepted and allow_memory_update:
                update = self.memory.add(
                    user_id, encoded["speaker_embedding"], audio_id
                )
                result["memory_initialized"] = bool(update["added"])
                result["memory_size"] = int(update["memory_size"])
            self._audit(result)
            return result

        neighbors = self.memory.search(
            user_id,
            encoded["speaker_embedding"],
            top_k=self.references,
        )
        reference_tensor = self._padded_references(neighbors.embeddings)
        output = self.attention_model(
            encoded["general_logits"],
            encoded["general_embedding"],
            encoded["speaker_embedding_tensor"],
            reference_tensor,
        )
        probabilities = torch.softmax(output["logits"], dim=-1)[0]
        personalized_bonafide = float(probabilities[0].cpu())
        personalized_spoof = float(probabilities[1].cpu())
        similarity = float(output["mean_similarity"][0].cpu())
        accepted = (
            personalized_spoof
            < self.calibration["personalized_spoof_threshold"]
            and general_bonafide
            >= self.calibration["minimum_general_bonafide_probability"]
            and similarity
            >= self.calibration["speaker_similarity_threshold"]
        )
        result.update(
            {
                "mode": "learned_personalized_verification",
                "personalized_bonafide_probability": personalized_bonafide,
                "personalized_spoof_probability": personalized_spoof,
                "speaker_similarity": similarity,
                "attention_gate": float(output["gate"][0].cpu()),
                "neighbor_audio_ids": neighbors.audio_ids,
                "final_prediction": 0 if accepted else 1,
                "decision": "bonafide_verified" if accepted else "spoof_or_mismatch",
            }
        )
        should_update = (
            accepted
            and allow_memory_update
            and general_bonafide
            >= self.calibration[
                "memory_update_general_bonafide_probability"
            ]
            and personalized_bonafide
            >= self.calibration[
                "memory_update_personalized_bonafide_probability"
            ]
        )
        if should_update:
            update = self.memory.add(
                user_id, encoded["speaker_embedding"], audio_id
            )
            result["memory_updated"] = bool(update["added"])
            result["memory_size"] = int(update["memory_size"])
            result["memory_update_reason"] = str(update["reason"])
        self._audit(result)
        return result
