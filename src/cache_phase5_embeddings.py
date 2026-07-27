"""Cache Phase 2/3 outputs used by the learned Phase 5 memory model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cached_dataset import CachedFeatureDataset
from src.config import PROJECT_ROOT, SETTINGS
from src.model_phase2 import Phase2GeneralDetector
from src.model_phase3 import Phase3SpeakerAwareDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "dev", "eval"), required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_models(device: torch.device):
    phase2_checkpoint = torch.load(
        PROJECT_ROOT / "checkpoints" / "phase2_best.pth",
        map_location=device,
    )
    general = Phase2GeneralDetector(**phase2_checkpoint["dimensions"]).to(device)
    general.load_state_dict(phase2_checkpoint["model_state_dict"])
    general.eval()

    phase3_checkpoint = torch.load(
        PROJECT_ROOT / "checkpoints" / "phase3_best.pth",
        map_location=device,
    )
    speaker = Phase3SpeakerAwareDetector(**phase3_checkpoint["dimensions"]).to(device)
    speaker.load_state_dict(phase3_checkpoint["model_state_dict"])
    speaker.eval()
    return general, speaker


def model_arguments(batch, device):
    return (
        batch["mfcc"].to(device),
        batch["lfcc"].to(device),
        batch["handcrafted"].to(device),
        batch["xlsr"].to(device),
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / "cache" / "phase5"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}_embeddings.npz"
    if output_path.exists() and not args.overwrite:
        print(f"Already exists: {output_path}")
        print("Use --overwrite to regenerate it.")
        return

    device = torch.device(SETTINGS.device)
    dataset = CachedFeatureDataset(args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    general, speaker = load_models(device)
    audio_ids, speaker_ids, labels = [], [], []
    general_logits, general_embeddings, speaker_embeddings = [], [], []

    for batch in tqdm(loader, desc=f"cache Phase 5 {args.split}"):
        arguments = model_arguments(batch, device)
        general_output = general(*arguments)
        speaker_output = speaker(*arguments)
        audio_ids.extend(map(str, batch["audio_id"]))
        speaker_ids.extend(map(str, batch["speaker_id"]))
        labels.extend(batch["label"].numpy().astype(np.int64))
        general_logits.append(general_output["logits"].cpu().numpy())
        general_embeddings.append(general_output["embedding"].cpu().numpy())
        speaker_embeddings.append(
            speaker_output["speaker_embedding"].cpu().numpy()
        )

    np.savez_compressed(
        output_path,
        audio_ids=np.asarray(audio_ids, dtype=str),
        speaker_ids=np.asarray(speaker_ids, dtype=str),
        labels=np.asarray(labels, dtype=np.int64),
        general_logits=np.concatenate(general_logits).astype(np.float32),
        general_embeddings=np.concatenate(general_embeddings).astype(np.float32),
        speaker_embeddings=np.concatenate(speaker_embeddings).astype(np.float32),
    )
    print(f"Saved {len(labels)} samples: {output_path}")


if __name__ == "__main__":
    main()
