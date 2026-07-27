"""Validate calibration and Phase 6 module imports."""

import json
from pathlib import Path

from src.config import PROJECT_ROOT
from src.model_phase5 import Phase5MemoryAttention
from src.online_detector_phase6 import Phase6OnlineDetector


calibration_path = PROJECT_ROOT / "checkpoints" / "phase6_calibration.json"
assert calibration_path.exists(), (
    "Run `python -m src.calibrate_phase6` before this smoke test."
)
calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
required = {
    "general_spoof_threshold",
    "personalized_spoof_threshold",
    "speaker_similarity_threshold",
    "initial_enrollment_bonafide_probability",
}
assert required <= set(calibration)
assert Phase5MemoryAttention is not None
assert Phase6OnlineDetector is not None
print("Phase 6 calibration and imports smoke test passed.")
