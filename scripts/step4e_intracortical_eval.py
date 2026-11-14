#!/usr/bin/env python3
"""Step 4e: decode reach direction from macaque M1/PMd spikes before the hand moves.

Window length is chosen by 5-fold cross-validation on the official NLB train
split only; the fitted decoder is then evaluated once on the held-out val split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bci.intracortical_decoder import DecoderWindow, ReachIntentDecoder
from bci.nlb_data import DEFAULT_NWB, SCHEME_TARGETS, load_mc_maze

DIRECTION_TARGETS = SCHEME_TARGETS["four"]


def accuracy(decoder, trials, indices):
    predictions = [decoder.predict_trial(trials, i)["intent"] for i in indices]
    truth = trials.labels[list(indices)]
    correct = np.array(predictions) == truth
    per_class = {c: float(correct[truth == c].mean()) for c in DIRECTION_TARGETS if (truth == c).any()}
    return float(correct.mean()), float(np.mean(list(per_class.values()))), predictions


def cross_validate(trials, train, window, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(train)
    scores = []
    for k in range(folds):
        held = order[k::folds]
        fit = np.setdiff1d(order, held)
        decoder = ReachIntentDecoder(DIRECTION_TARGETS, window).fit(trials, fit)
        scores.append(accuracy(decoder, trials, held)[1])
    return float(np.mean(scores))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nwb", type=Path, default=DEFAULT_NWB)
    parser.add_argument("--scheme", choices=sorted(SCHEME_TARGETS), default="four")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    global DIRECTION_TARGETS
    DIRECTION_TARGETS = SCHEME_TARGETS[args.scheme]
    args.output = args.output or Path("artifacts/results/step4e_intracortical_eval.json" if args.scheme == "four"
                                      else f"artifacts/results/step4f_intracortical_eval_{args.scheme}.json")
    trials = load_mc_maze(args.nwb, args.scheme)
    train = np.flatnonzero(trials.split == "train")
    val = np.flatnonzero(trials.split == "val")
    print(f"MC_Maze_Large: {trials.n_units} units, {len(train)} train / {len(val)} val trials (official NLB split)")

    candidates = [DecoderWindow(-length, 0.0, 0.1) for length in (0.2, 0.3, 0.4, 0.5)]
    cv = {c.start: cross_validate(trials, train, c) for c in candidates}
    window = max(candidates, key=lambda c: cv[c.start])
    print("5-fold CV on train (balanced accuracy): " + ", ".join(f"[{c.start:+.1f}, 0] s {cv[c.start]:.1%}" for c in candidates))

    decoder = ReachIntentDecoder(DIRECTION_TARGETS, window).fit(trials, train)
    acc, balanced, predictions = accuracy(decoder, trials, val)
    truth = trials.labels[val]
    confusion = {t: {p: int(np.sum((truth == t) & (np.array(predictions) == p))) for p in DIRECTION_TARGETS}
                 for t in DIRECTION_TARGETS}
    majority = max((truth == c).mean() for c in DIRECTION_TARGETS)
    print(f"Held-out val, window [{window.start:+.1f}, 0] s before movement onset:")
    print(f"  accuracy {acc:.1%}, balanced {balanced:.1%} (majority-class chance {majority:.1%})")
    for t, row in confusion.items():
        print(f"    {t:>6}: " + "  ".join(f"{p}={n}" for p, n in row.items()))

    rng = np.random.default_rng(1)
    shuffled = trials.__class__(**{**trials.__dict__, "labels": trials.labels.copy()})
    shuffled.labels[train] = rng.permutation(trials.labels[train])
    control = ReachIntentDecoder(DIRECTION_TARGETS, window).fit(shuffled, train)
    control_acc = accuracy(control, trials, val)[0]
    print(f"  shuffled-label control: {control_acc:.1%}")

    course = []
    for end in np.round(np.arange(-0.6, 0.25, 0.05), 2):
        w = DecoderWindow(end - 0.2, end, 0.1)
        course.append({"window_end": float(end), "accuracy": accuracy(
            ReachIntentDecoder(DIRECTION_TARGETS, w).fit(trials, train), trials, val)[0]})
    print("  time course (200 ms window ending at t relative to movement onset):")
    print("    " + "  ".join(f"{c['window_end']:+.2f}:{c['accuracy']:.0%}" for c in course))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "dataset": "NLB MC_Maze_Large (DANDI 000138), monkey Jenkins, M1/PMd", "scheme": args.scheme,
        "classes": list(DIRECTION_TARGETS), "units": trials.n_units,
        "train_trials": int(len(train)), "val_trials": int(len(val)),
        "window": {"start": window.start, "stop": window.stop, "bin_seconds": window.bin_seconds},
        "cv_balanced_accuracy": {str(k): v for k, v in cv.items()},
        "val": {"accuracy": acc, "balanced_accuracy": balanced, "majority_chance": float(majority),
                "confusion": confusion, "shuffled_label_control": control_acc},
        "time_course": course,
    }, indent=2))
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
