"""Basic Phase 8 protocol and subset smoke test."""

from pathlib import Path
import tempfile

from src.phase8_data import (
    parse_asvspoof2021_la_protocol,
    select_reproducible_subset,
)


content = """\
LA_0001 LA_E_0000001 codec tx - bonafide notrim eval
LA_0001 LA_E_0000002 codec tx A01 spoof notrim eval
LA_0002 LA_E_0000003 codec tx - bonafide notrim eval
LA_0002 LA_E_0000004 codec tx A02 spoof notrim eval
"""
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "keys.txt"
    path.write_text(content, encoding="utf-8")
    frame = parse_asvspoof2021_la_protocol(path)
    subset = select_reproducible_subset(frame, 4, True, 42)
    assert len(subset) == 4
    assert subset["label"].value_counts().to_dict() == {0: 2, 1: 2}
print("Phase 8 protocol smoke test passed.")
