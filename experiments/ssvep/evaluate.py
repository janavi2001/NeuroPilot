#!/usr/bin/env python3
"""Evaluate CCA and FBCCA on real BETA SSVEP trials."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .dataset import crop_window, load_beta_subject, select_channels
from .decoder import CCAConfig, FBCCAConfig, SSVEPDecoder
from .preprocessing import prepare_eeg


def target_subset(all_frequencies: np.ndarray, count: int) -> np.ndarray:
    if count < 1 or count > len(all_frequencies):
        raise ValueError("Target count is outside the available dataset classes")
    indices = np.linspace(0, len(all_frequencies) - 1, count, dtype=int)
    return all_frequencies[np.unique(indices)]


def evaluate(subject: int, target_count: int, window_seconds: float, channels: tuple[str, ...] | None = None) -> dict:
    dataset = load_beta_subject(subject)
    available = np.unique(dataset.frequencies_hz)
    frequencies = target_subset(available, target_count)
    keep = np.isin(dataset.frequencies_hz, frequencies)
    data = crop_window(dataset, window_seconds)[keep]
    labels = dataset.frequencies_hz[keep]
    channel_indices = select_channels(dataset, channels)
    data = prepare_eeg(data, channel_indices)
    decoders = {
        "CCA": SSVEPDecoder(frequencies, CCAConfig(window_seconds=window_seconds)),
        "FBCCA": SSVEPDecoder(frequencies, FBCCAConfig(window_seconds=window_seconds)),
    }
    result = {"subject": subject, "targets": target_count, "window_seconds": window_seconds, "trials": len(labels)}
    for name, decoder in decoders.items():
        predictions, runtimes = [], []
        for trial in data:
            start = time.perf_counter()
            predictions.append(decoder.predict(trial).frequency_hz)
            runtimes.append(time.perf_counter() - start)
        predictions = np.asarray(predictions)
        result[name] = {
            "correct": int(np.sum(predictions == labels)),
            "accuracy": float(np.mean(predictions == labels)),
            "mean_runtime_seconds": float(np.mean(runtimes)),
            "confusion_matrix": _confusion(labels, predictions, frequencies),
        }
    return result


def _confusion(labels, predictions, frequencies):
    return [[int(np.sum((labels == actual) & (predictions == predicted))) for predicted in frequencies] for actual in frequencies]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--targets", type=int, nargs="+", default=[4, 8, 12, 20, 40])
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path("experiments/ssvep/artifacts/results/evaluation.json"))
    args = parser.parse_args()
    results = [evaluate(args.subject, count, args.window) for count in args.targets]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    for result in results:
        print(f"Targets={result['targets']} trials={result['trials']}")
        for decoder in ("CCA", "FBCCA"):
            item = result[decoder]
            print(f"  {decoder}: {item['correct']}/{result['trials']} = {item['accuracy']:.3f}; runtime={item['mean_runtime_seconds']:.6f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
