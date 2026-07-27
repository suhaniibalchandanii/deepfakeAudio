"""Apply frozen ASVspoof2019 Phase 2/3 models to 2021 feature caches."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import PROJECT_ROOT, SETTINGS
from src.model_phase2 import Phase2GeneralDetector
from src.model_phase3 import Phase3SpeakerAwareDetector
from src.phase8_data import Phase8CachedFeatureDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_models(device):
    phase2 = torch.load(
        PROJECT_ROOT / "checkpoints" / "phase2_best.pth",
        map_location=device,
    )
    general = Phase2GeneralDetector(**phase2["dimensions"]).to(device)
    general.load_state_dict(phase2["model_state_dict"])
    general.eval()

    phase3 = torch.load(
        PROJECT_ROOT / "checkpoints" / "phase3_best.pth",
        map_location=device,
    )
    speaker = Phase3SpeakerAwareDetector(**phase3["dimensions"]).to(device)
    speaker.load_state_dict(phase3["model_state_dict"])
    speaker.eval()
    return general, speaker


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / "cache" / "phase8_2021"
    output_path = output_dir / "la_eval_embeddings.npz"
    if output_path.exists() and not args.overwrite:
        print(f"Already exists: {output_path}")
        return
    dataset = Phase8CachedFeatureDataset(output_dir / "la_eval")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    device = torch.device(SETTINGS.device)
    general, speaker = load_models(device)
    collected = {
        "audio_ids": [],
        "speaker_ids": [],
        "labels": [],
        "general_logits": [],
        "general_embeddings": [],
        "speaker_embeddings": [],
    }
    exclusions = []

    for batch in tqdm(loader, desc="Frozen 2019 models → 2021"):
        arguments = (
            batch["mfcc"].to(device),
            batch["lfcc"].to(device),
            batch["handcrafted"].to(device),
            batch["xlsr"].to(device),
        )
        general_output = general(*arguments)
        speaker_output = speaker(*arguments)
        arrays = {
            "general_logits": general_output["logits"].cpu().numpy(),
            "general_embeddings": general_output["embedding"].cpu().numpy(),
            "speaker_embeddings": speaker_output[
                "speaker_embedding"
            ].cpu().numpy(),
        }
        valid = np.logical_and.reduce(
            [np.isfinite(array).all(axis=1) for array in arrays.values()]
        )
        for index, is_valid in enumerate(valid):
            if not is_valid:
                exclusions.append(
                    {
                        "audio_id": str(batch["audio_id"][index]),
                        "reason": "non-finite frozen-model output",
                    }
                )
        collected["audio_ids"].extend(
            np.asarray(batch["audio_id"], dtype=str)[valid].tolist()
        )
        collected["speaker_ids"].extend(
            np.asarray(batch["speaker_id"], dtype=str)[valid].tolist()
        )
        collected["labels"].extend(batch["label"].numpy()[valid].tolist())
        for key, array in arrays.items():
            collected[key].append(array[valid].astype(np.float32))

    np.savez_compressed(
        output_path,
        audio_ids=np.asarray(collected["audio_ids"], dtype=str),
        speaker_ids=np.asarray(collected["speaker_ids"], dtype=str),
        labels=np.asarray(collected["labels"], dtype=np.int64),
        general_logits=np.concatenate(collected["general_logits"]),
        general_embeddings=np.concatenate(collected["general_embeddings"]),
        speaker_embeddings=np.concatenate(collected["speaker_embeddings"]),
    )
    pd.DataFrame(exclusions).to_csv(
        output_dir / "excluded_samples.csv", index=False
    )
    print(f"Saved samples: {len(collected['labels'])}")
    print(f"Excluded: {len(exclusions)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
