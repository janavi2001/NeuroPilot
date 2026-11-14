#!/usr/bin/env python3
"""Step 4c: measure single-look versus two-look-confirm selection on held-out BETA subjects.

Reads the per-trial decoder outputs saved by scripts/step4b_ssvep_eval.py (the
frequency set and UNSURE threshold were chosen on S11-S20; these are S1-S10).
Each target has four recorded trials per subject; trials 1+2 and 3+4 form the
two-look pairs, so no trial is reused.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def wolpaw_bits(accuracy: float, targets: int) -> float:
    """Information transfer per selection (Wolpaw et al. 2000), forced choice."""
    if accuracy >= 1.0:
        return math.log2(targets)
    if accuracy <= 1.0 / targets:
        return 0.0
    return math.log2(targets) + accuracy * math.log2(accuracy) + (1 - accuracy) * math.log2((1 - accuracy) / (targets - 1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, default=Path("artifacts/results/step4b_ssvep_eval.json"))
    parser.add_argument("--seconds-per-look", type=float, default=2.5, help="2.0 s stimulation + 0.5 s gaze shift (BETA cue)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/step4c_confirm_eval.json"))
    args = parser.parse_args()
    if not args.eval.is_file():
        parser.error(f"{args.eval} not found; run scripts/step4b_ssvep_eval.py first")
    saved = json.loads(args.eval.read_text())
    threshold = saved["margin_threshold"]
    targets = len(saved["config"]["targets"])
    rows = saved["trials"]["test"]

    single = {"looks": 0, "actions": 0, "wrong": 0}
    for row in rows:
        single["looks"] += 1
        if row["margin"] >= threshold:
            single["actions"] += 1
            single["wrong"] += row["best_target"] != row["ground_truth"]

    by_subject_target = defaultdict(list)
    for row in rows:
        by_subject_target[(row["subject"], row["ground_truth"])].append(row)
    confirm = {"pairs": 0, "actions": 0, "wrong": 0}
    for trials in by_subject_target.values():
        trials.sort(key=lambda row: row["trial_index"])
        for first, second in zip(trials[0::2], trials[1::2]):
            confirm["pairs"] += 1
            agree = first["best_target"] == second["best_target"]
            accepted = first["margin"] >= threshold and second["margin"] >= threshold
            if agree and accepted:
                confirm["actions"] += 1
                confirm["wrong"] += first["best_target"] != first["ground_truth"]

    raw_accuracy = saved["test_without_threshold"]["raw_accuracy"]
    bits = wolpaw_bits(raw_accuracy, targets)
    result = {
        "held_out_subjects": sorted({row["subject"] for row in rows}),
        "margin_threshold": threshold,
        "single_look": {
            **single,
            "wrong_action_rate": single["wrong"] / single["looks"],
            "no_action_rate": 1 - single["actions"] / single["looks"],
        },
        "two_look_confirm": {
            **confirm,
            "wrong_action_rate": confirm["wrong"] / confirm["pairs"],
            "no_action_rate": 1 - confirm["actions"] / confirm["pairs"],
        },
        "itr": {
            "raw_accuracy": raw_accuracy,
            "targets": targets,
            "bits_per_selection": bits,
            "seconds_per_look": args.seconds_per_look,
            "bits_per_minute": bits * 60 / args.seconds_per_look,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    s, c, i = result["single_look"], result["two_look_confirm"], result["itr"]
    print(f"Held-out subjects {result['held_out_subjects']}, UNSURE threshold {threshold}")
    print(f"Single look:      {s['looks']} looks, wrong action {s['wrong_action_rate']:.1%}, no action {s['no_action_rate']:.1%}")
    print(f"Two-look confirm: {c['pairs']} pairs, wrong action {c['wrong_action_rate']:.1%} ({c['wrong']}/{c['pairs']}), "
          f"no action {c['no_action_rate']:.1%}")
    print(f"ITR (single look, forced choice): {i['bits_per_selection']:.2f} bits/selection, "
          f"{i['bits_per_minute']:.1f} bits/min at {args.seconds_per_look:.1f} s/look")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
