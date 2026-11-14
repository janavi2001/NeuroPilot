#!/usr/bin/env python3
"""Step 4b: calibrate the UNSURE margin on some BETA subjects, test on others."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bci.ssvep_data import load_beta_trials
from bci.ssvep_decoder import SSVEPDecoder
from interface.ssvep_selector import DEFAULT_CONFIG, load_ssvep_config


def decode_subject(subject: int, config) -> list[dict]:
    eeg, labels, _ = load_beta_trials(subject, config.channels, config.window_seconds, list(config.targets.values()))
    decoder = SSVEPDecoder(config.targets, len(config.channels), window_seconds=config.window_seconds, harmonics=config.harmonics)
    frequency_to_name = {frequency: name for name, frequency in config.targets.items()}
    rows = []
    for index, (window, label) in enumerate(zip(eeg, labels)):
        start = time.perf_counter()
        prediction = decoder.predict(window)
        rows.append({
            "subject": subject,
            "trial_index": index,
            "ground_truth": frequency_to_name[float(label)],
            "best_target": prediction["best_target"],
            "margin": prediction["margin"],
            "runtime_seconds": time.perf_counter() - start,
        })
    return rows


def score(rows: list[dict], threshold: float, names: list[str]) -> dict:
    correct = np.array([row["best_target"] == row["ground_truth"] for row in rows])
    accepted = np.array([row["margin"] >= threshold for row in rows])
    confusion = {truth: {name: 0 for name in names + ["UNSURE"]} for truth in names}
    for row, keep in zip(rows, accepted):
        confusion[row["ground_truth"]][row["best_target"] if keep else "UNSURE"] += 1
    per_subject = {}
    for subject in sorted({row["subject"] for row in rows}):
        mask = np.array([row["subject"] == subject for row in rows])
        per_subject[subject] = {
            "raw_accuracy": float(correct[mask].mean()),
            "rejected_fraction": float(1 - accepted[mask].mean()),
            "wrong_action_fraction": float((accepted & ~correct)[mask].mean()),
        }
    return {
        "trials": len(rows),
        "raw_accuracy": float(correct.mean()),
        "rejected_fraction": float(1 - accepted.mean()),
        "accepted_accuracy": float(correct[accepted].mean()) if accepted.any() else None,
        "wrong_action_fraction": float((accepted & ~correct).mean()),
        "per_subject": per_subject,
        "confusion": confusion,
        "mean_runtime_ms": 1000 * float(np.mean([row["runtime_seconds"] for row in rows])),
    }


def calibrate(rows: list[dict], target_accuracy: float, names: list[str]) -> float:
    for threshold in np.round(np.arange(0.0, 0.3, 0.005), 3):
        result = score(rows, float(threshold), names)
        if result["accepted_accuracy"] is not None and result["accepted_accuracy"] >= target_accuracy:
            return float(threshold)
    raise RuntimeError(f"No margin threshold below 0.3 reaches {target_accuracy:.0%} accepted accuracy")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calibration-subjects", type=int, nargs="+", default=list(range(11, 21)))
    parser.add_argument("--test-subjects", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--target-accuracy", type=float, default=0.95, help="Accepted-trial accuracy the threshold must reach on calibration subjects")
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/step4b_ssvep_eval.json"))
    args = parser.parse_args()
    if set(args.calibration_subjects) & set(args.test_subjects):
        parser.error("calibration and test subjects must not overlap")

    config = load_ssvep_config(args.config)
    names = list(config.targets)
    calibration_rows = [row for subject in args.calibration_subjects for row in decode_subject(subject, config)]
    test_rows = [row for subject in args.test_subjects for row in decode_subject(subject, config)]
    threshold = calibrate(calibration_rows, args.target_accuracy, names)
    calibration = score(calibration_rows, threshold, names)
    test = score(test_rows, threshold, names)
    test_no_threshold = score(test_rows, 0.0, names)

    print(f"Targets: {config.targets}  window={config.window_seconds}s  channels={','.join(config.channels)}")
    print(f"Calibration subjects {args.calibration_subjects}: margin threshold {threshold:.3f} "
          f"(accepted accuracy {calibration['accepted_accuracy']:.1%}, rejected {calibration['rejected_fraction']:.1%})")
    print(f"Held-out subjects {args.test_subjects}, {test['trials']} trials:")
    print(f"  raw accuracy (no UNSURE)  {test_no_threshold['raw_accuracy']:.1%}")
    print(f"  rejected as UNSURE        {test['rejected_fraction']:.1%}")
    print(f"  accuracy when accepted    {test['accepted_accuracy']:.1%}")
    print(f"  wrong action (of all)     {test['wrong_action_fraction']:.1%}")
    print(f"  mean decode time          {test['mean_runtime_ms']:.2f} ms")
    print("  per subject: " + ", ".join(
        f"S{s} raw {v['raw_accuracy']:.0%}/unsure {v['rejected_fraction']:.0%}/wrong {v['wrong_action_fraction']:.0%}"
        for s, v in test["per_subject"].items()))
    print("  confusion (rows = ground truth):")
    for truth, row in test["confusion"].items():
        print(f"    {truth:>7}: " + "  ".join(f"{name}={count}" for name, count in row.items()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "config": {"targets": config.targets, "channels": config.channels, "window_seconds": config.window_seconds},
        "target_accuracy": args.target_accuracy,
        "margin_threshold": threshold,
        "calibration": calibration,
        "test": test,
        "test_without_threshold": test_no_threshold,
        "trials": {"calibration": calibration_rows, "test": test_rows},
    }, indent=2))
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
