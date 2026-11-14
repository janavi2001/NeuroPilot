#!/usr/bin/env python3
"""Step 4d: choose an SSVEP early-stopping rule on S11-S20, report on held-out S1-S10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bci.metrics import wolpaw_bits
from bci.ssvep_data import load_beta_trials
from bci.ssvep_stopping import StoppingRule, margins_over_time
from interface.ssvep_selector import DEFAULT_CONFIG, load_ssvep_config

GAZE_SHIFT_SECONDS = 0.5  # BETA cue period; counted in every look's duration


def trial_curves(subjects, config):
    """Per trial: true index, best index and margin at each checkpoint, and at the full window."""
    names = list(config.targets)
    times = [round(t, 1) for t in np.arange(0.3, config.window_seconds - 1e-9, 0.1)] + [config.window_seconds]
    rows = []
    for subject in subjects:
        eeg, labels, meta = load_beta_trials(subject, config.channels, config.window_seconds, list(config.targets.values()))
        to_name = {f: n for n, f in config.targets.items()}
        for window, label in zip(eeg, labels):
            curve, _ = margins_over_time(window, config.targets, meta["sampling_rate_hz"], times, config.harmonics)
            rows.append({"subject": subject, "truth": names.index(to_name[float(label)]),
                         "best": curve[0].astype(int), "margin": curve[1]})
    return rows, times


def decide(row, times, rule):
    """Checkpoint index where ``rule`` stops (the last index means the full window)."""
    for i, t in enumerate(times[:-1]):
        if t >= rule.min_seconds - 1e-9 and row["margin"][i] >= rule.required_margin(t, times[-1]):
            return i
    return len(times) - 1


def simulate(rows, times, rule, unsure_threshold, targets):
    correct, accepted, durations = [], [], []
    for row in rows:
        index = decide(row, times, rule)
        early = index < len(times) - 1
        ok = row["best"][index] == row["truth"]
        accept = early or row["margin"][index] >= unsure_threshold
        correct.append(ok)
        accepted.append(accept)
        durations.append(times[index])
    correct, accepted, durations = map(np.asarray, (correct, accepted, durations))
    accuracy = float(correct.mean())
    mean_seconds = float(durations.mean())
    bits = wolpaw_bits(accuracy, targets)
    return {
        "rule": {"min_seconds": rule.min_seconds, "base_margin": rule.base_margin, "exponent": rule.exponent},
        "trials": len(rows),
        "forced_accuracy": accuracy, "wrong_action_rate": float((accepted & ~correct).mean()),
        "unsure_rate": float(1 - accepted.mean()), "mean_decision_seconds": mean_seconds,
        "early_fraction": float((durations < times[-1]).mean()),
        "bits_per_selection": bits, "bits_per_minute": bits * 60 / (mean_seconds + GAZE_SHIFT_SECONDS),
    }


def confirm_pairs(rows, times, rule, unsure_threshold):
    """Two-look confirm on consecutive trials of the same subject and target (trials 1+2, 3+4)."""
    groups = {}
    for row in rows:
        groups.setdefault((row["subject"], row["truth"]), []).append(row)
    pairs = actions = wrong = 0
    seconds = []
    for trials in groups.values():
        for first, second in zip(trials[0::2], trials[1::2]):
            looks = []
            for row in (first, second):
                index = decide(row, times, rule)
                accept = index < len(times) - 1 or row["margin"][index] >= unsure_threshold
                looks.append((row["best"][index], accept, times[index]))
            pairs += 1
            seconds.append(looks[0][2] + looks[1][2])
            if looks[0][1] and looks[1][1] and looks[0][0] == looks[1][0]:
                actions += 1
                wrong += int(looks[0][0] != first["truth"])
    return {"pairs": pairs, "wrong": wrong, "wrong_action_rate": wrong / pairs,
            "no_action_rate": 1 - actions / pairs, "mean_seconds_two_looks": float(np.mean(seconds))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calibration-subjects", type=int, nargs="+", default=list(range(11, 21)))
    parser.add_argument("--test-subjects", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/step4d_early_stop_eval.json"))
    args = parser.parse_args()
    if set(args.calibration_subjects) & set(args.test_subjects):
        parser.error("calibration and test subjects must not overlap")
    config = load_ssvep_config(args.config)
    n = len(config.targets)
    calibration, times = trial_curves(args.calibration_subjects, config)
    test, _ = trial_curves(args.test_subjects, config)

    fixed = StoppingRule(config.window_seconds, float("inf"))
    baseline_cal = simulate(calibration, times, fixed, config.margin_threshold, n)
    grid = [StoppingRule(m, float(a), k) for m in (0.3, 0.5, 0.7) for k in (0.0, 0.5, 1.0, 1.5, 2.0)
            for a in np.round(np.arange(0.05, 0.5, 0.01), 2)]
    candidates = [(rule, simulate(calibration, times, rule, config.margin_threshold, n)) for rule in grid]
    allowed = [c for c in candidates if c[1]["wrong_action_rate"] <= baseline_cal["wrong_action_rate"] + 1e-12]
    rule, chosen = max(allowed, key=lambda c: c[1]["bits_per_minute"])
    baseline_test = simulate(test, times, fixed, config.margin_threshold, n)
    result_test = simulate(test, times, rule, config.margin_threshold, n)
    confirm_fixed = confirm_pairs(test, times, fixed, config.margin_threshold)
    confirm_early = confirm_pairs(test, times, rule, config.margin_threshold)

    def show(label, r):
        print(f"  {label:<24} acc {r['forced_accuracy']:.1%}  wrong {r['wrong_action_rate']:.1%}  "
              f"unsure {r['unsure_rate']:.1%}  mean time {r['mean_decision_seconds']:.2f} s  "
              f"early {r['early_fraction']:.0%}  {r['bits_per_minute']:.1f} bits/min")

    print(f"Calibration S{args.calibration_subjects[0]}-S{args.calibration_subjects[-1]}: "
          f"rule: stop when margin >= {rule.base_margin} * (2 s / t)^{rule.exponent}, t >= {rule.min_seconds} s "
          f"(wrong-action cap {baseline_cal['wrong_action_rate']:.1%})")
    show("calibration fixed 2 s", baseline_cal)
    show("calibration early stop", chosen)
    print(f"Held-out S{args.test_subjects[0]}-S{args.test_subjects[-1]} ({len(test)} trials):")
    show("fixed 2 s", baseline_test)
    show("early stop", result_test)
    for label, c in (("two-look confirm, fixed", confirm_fixed), ("two-look confirm, early", confirm_early)):
        print(f"  {label:<24} wrong {c['wrong_action_rate']:.1%} ({c['wrong']}/{c['pairs']})  "
              f"no action {c['no_action_rate']:.1%}  mean time for both looks {c['mean_seconds_two_looks']:.2f} s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rule": chosen["rule"], "gaze_shift_seconds": GAZE_SHIFT_SECONDS,
                                       "calibration": {"fixed": baseline_cal, "early": chosen},
                                       "test": {"fixed": baseline_test, "early": result_test,
                                                "confirm_fixed": confirm_fixed, "confirm_early": confirm_early}},
                                      indent=2))
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
