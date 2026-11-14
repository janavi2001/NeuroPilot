#!/usr/bin/env python3
"""Multi-subject CCA sweep over target count, window length, and channel set.

Each subject is loaded once; every (targets, window, channels) configuration is
scored on the same real BETA trials. Per-trial predictions and margins are kept
so accuracy/rejection trade-offs can be reported without re-decoding.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .dataset import crop_window, load_beta_subject, select_channels
from .decoder import CCAConfig, SSVEPDecoder
from .evaluate import target_subset
from .preprocessing import prepare_eeg

CHANNEL_SETS = {
    "occipital3": ("O1", "Oz", "O2"),
    "posterior8": ("O1", "Oz", "O2", "PO3", "PO4", "PO7", "PO8", "POz"),
}
MARGIN_THRESHOLDS = (0.0, 0.02, 0.05, 0.08, 0.1)


def sweep_subject(subject: int, target_counts, windows, channel_sets) -> list[dict]:
    dataset = load_beta_subject(subject)
    available = np.unique(dataset.frequencies_hz)
    rows = []
    for window in windows:
        cropped = crop_window(dataset, window)
        for channel_name in channel_sets:
            data = prepare_eeg(cropped, select_channels(dataset, CHANNEL_SETS[channel_name]))
            for count in target_counts:
                frequencies = target_subset(available, count)
                keep = np.isin(dataset.frequencies_hz, frequencies)
                decoder = SSVEPDecoder(frequencies, CCAConfig(window_seconds=window))
                labels = dataset.frequencies_hz[keep]
                predictions, margins, runtimes = [], [], []
                for trial in data[keep]:
                    start = time.perf_counter()
                    selection = decoder.predict(trial)
                    runtimes.append(time.perf_counter() - start)
                    predictions.append(selection.frequency_hz)
                    margins.append(selection.score_margin)
                rows.append({
                    "subject": subject,
                    "targets": int(len(frequencies)),
                    "frequencies_hz": [float(f) for f in frequencies],
                    "window_seconds": window,
                    "channels": channel_name,
                    "labels": [float(x) for x in labels],
                    "predictions": [float(x) for x in predictions],
                    "margins": [float(x) for x in margins],
                    "mean_runtime_seconds": float(np.mean(runtimes)),
                })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Pool trials across subjects per configuration; report accuracy vs. rejection."""

    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["targets"], row["window_seconds"], row["channels"]), []).append(row)
    summary = []
    for (targets, window, channels), items in sorted(groups.items()):
        labels = np.concatenate([item["labels"] for item in items])
        predictions = np.concatenate([item["predictions"] for item in items])
        margins = np.concatenate([item["margins"] for item in items])
        correct = predictions == labels
        per_subject = {item["subject"]: float(np.mean(np.asarray(item["predictions"]) == np.asarray(item["labels"]))) for item in items}
        thresholds = []
        for threshold in MARGIN_THRESHOLDS:
            accepted = margins >= threshold
            thresholds.append({
                "margin_threshold": threshold,
                "rejected_fraction": float(1 - accepted.mean()),
                "accepted_accuracy": float(correct[accepted].mean()) if accepted.any() else None,
            })
        summary.append({
            "targets": targets,
            "window_seconds": window,
            "channels": channels,
            "trials": int(len(labels)),
            "accuracy": float(correct.mean()),
            "worst_subject_accuracy": min(per_subject.values()),
            "per_subject_accuracy": per_subject,
            "mean_runtime_seconds": float(np.mean([item["mean_runtime_seconds"] for item in items])),
            "rejection": thresholds,
        })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--targets", type=int, nargs="+", default=[4, 6, 10, 20])
    parser.add_argument("--windows", type=float, nargs="+", default=[1.5, 2.0])
    parser.add_argument("--channels", nargs="+", default=list(CHANNEL_SETS), choices=list(CHANNEL_SETS))
    parser.add_argument("--output", type=Path, default=Path("experiments/ssvep/artifacts/results/sweep.json"))
    args = parser.parse_args()

    rows = []
    for subject in args.subjects:
        print(f"Subject S{subject}...", flush=True)
        rows.extend(sweep_subject(subject, args.targets, args.windows, args.channels))
    summary = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "trials": rows}, indent=2))

    print(f"\n{'targets':>7} {'window':>6} {'channels':>11} {'trials':>6} {'acc':>6} {'worst':>6}  rejected/accuracy at margin>=0.05")
    for item in summary:
        at = next(t for t in item["rejection"] if t["margin_threshold"] == 0.05)
        print(
            f"{item['targets']:>7} {item['window_seconds']:>6.1f} {item['channels']:>11} {item['trials']:>6} "
            f"{item['accuracy']:>6.1%} {item['worst_subject_accuracy']:>6.1%}  "
            f"{at['rejected_fraction']:.1%} / {at['accepted_accuracy']:.1%}"
        )
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
