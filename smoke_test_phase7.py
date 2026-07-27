"""Smoke-test Phase 7 metric helpers."""

import numpy as np

from src.evaluate_phase7 import calculate_metrics, equal_error_rate


labels = np.asarray([0, 0, 1, 1])
scores = np.asarray([0.05, 0.20, 0.80, 0.95])
predictions = (scores >= 0.5).astype(int)
metrics = calculate_metrics(labels, predictions, scores)
assert metrics["accuracy"] == 1.0
assert metrics["macro_f1"] == 1.0
assert metrics["roc_auc"] == 1.0
assert equal_error_rate(labels, scores)["eer"] == 0.0
gated = calculate_metrics(labels, predictions, None)
assert gated["roc_auc"] is None
assert gated["eer"] is None
print("Phase 7 metric smoke test passed.")
