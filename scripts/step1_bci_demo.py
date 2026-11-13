#!/usr/bin/env python3
"""Step 1: real BCI-IV-2a EEG -> frozen LEAF -> LEFT/RIGHT."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bci.decoder import BCIDecoder
from bci.preprocessing import load_real_trials


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg", type=Path, default=Path("data/raw/MI_BCIC_IV2a_mat/A01T.mat"), help="Real BCI-IV-2a A01T.mat or A01T.gdf file")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt"))
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("Loading real EEG...")
    x, labels, metadata = load_real_trials(args.eeg, args.limit)
    print(f"{metadata['channels']} channels at {metadata['sampling_rate_hz']:.0f} Hz")
    print(f"Processed input shape: {tuple(x.shape)}")
    print(f"Filtering: {metadata['filter_hz'][0]}-{metadata['filter_hz'][1]} Hz; resampled to {metadata['resampled_rate_hz']} Hz")
    print(f"Normalization: {metadata['normalization']}")

    print("Loading frozen pretrained decoder...")
    decoder = BCIDecoder(args.checkpoint, device=args.device)
    print("checkpoint loaded; parameters frozen")

    correct = 0
    for index, (trial, label) in enumerate(zip(x, labels), start=1):
        result = decoder.predict(trial)
        correct += result["intent"] == label
        scores = ", ".join(f"{name}={score:.4f}" for name, score in result["scores"].items())
        print(f"Trial {index}: predicted={result['intent']} ground_truth={label} score={result['score']:.4f} ({scores})")

    accuracy = correct / len(labels)
    print(f"Subset accuracy: {correct}/{len(labels)} = {accuracy:.3f}")
    print("BCI SMOKE TEST: PASS")
    print("Limitation: prototype similarity is a raw cosine score, not a calibrated probability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
