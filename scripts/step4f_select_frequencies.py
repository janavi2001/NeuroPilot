#!/usr/bin/env python3
"""Step 4f: choose six SSVEP frequencies on calibration subjects S11-S20 only.

Candidates are every six-frequency subset of BETA's 8.0-15.8 Hz grid whose 1st-3rd
harmonics keep the largest achievable minimum separation (1.2 Hz). Each candidate
is scored by fixed-window CCA accuracy on S11-S20; held-out S1-S10 are not used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bci.ssvep_data import load_beta_trials
from bci.ssvep_decoder import SSVEPDecoder

CHANNELS = ("O1", "Oz", "O2", "PO3", "PO4", "PO7", "PO8", "POz")
GRID = [round(8.0 + 0.2 * i, 1) for i in range(40)]


def harmonic_gap(a: float, b: float, harmonics=(1, 2, 3)) -> float:
    return min(abs(a * h - b * k) for h in harmonics for k in harmonics)


def candidate_sets(size: int, min_gap: float) -> list[tuple[float, ...]]:
    found = []

    def extend(start, chosen):
        if len(chosen) == size:
            found.append(tuple(chosen))
            return
        for i in range(start, len(GRID)):
            if all(harmonic_gap(GRID[i], c) >= min_gap - 1e-9 for c in chosen):
                extend(i + 1, chosen + [GRID[i]])

    extend(0, [])
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(11, 21)))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/step4f_frequency_selection.json"))
    args = parser.parse_args()
    min_gap = next(g for g in np.arange(2.0, 0.0, -0.2) if candidate_sets(args.size, g))
    candidates = candidate_sets(args.size, min_gap)
    evenly = tuple(GRID[i] for i in np.linspace(0, 39, args.size).astype(int))
    if evenly not in candidates:
        candidates.append(evenly)
    print(f"{len(candidates) - 1} sets with min harmonic gap {min_gap:.1f} Hz (+ evenly spaced baseline {evenly})")

    data = [load_beta_trials(s, CHANNELS, 2.0) for s in args.subjects]
    scores = {}
    for frequencies in candidates:
        decoder = SSVEPDecoder({f"T{f}": f for f in frequencies}, len(CHANNELS), 250.0, 2.0, 3)
        correct = total = 0
        for eeg, labels, _ in data:
            keep = np.isin(labels, frequencies)
            for window, label in zip(eeg[keep], labels[keep]):
                correct += decoder.predict(window)["frequency_hz"] == label
                total += 1
        scores[frequencies] = correct / total
    ranked = sorted(scores, key=scores.get, reverse=True)
    for frequencies in ranked[:5]:
        print(f"  {frequencies}: {scores[frequencies]:.1%}")
    print(f"  evenly spaced baseline {evenly}: {scores[evenly]:.1%} (harmonic gap {min(harmonic_gap(a, b) for i, a in enumerate(evenly) for b in evenly[i+1:]):.1f} Hz)")
    best = ranked[0]
    print(f"Chosen: {best} ({scores[best]:.1%} on calibration S{args.subjects[0]}-S{args.subjects[-1]})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"min_harmonic_gap_hz": min_gap, "subjects": args.subjects, "chosen": best,
                                       "scores": {str(k): v for k, v in scores.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
