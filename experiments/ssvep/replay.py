#!/usr/bin/env python3
"""Replay one real BETA SSVEP trial and print target scores."""

from __future__ import annotations

import argparse

from .dataset import crop_window, load_beta_subject, select_channels
from .decoder import FBCCAConfig, SSVEPDecoder
from .preprocessing import prepare_eeg
from .visualization import save_score_plot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--plot", default="", help="Optional output PNG path")
    args = parser.parse_args()
    dataset = load_beta_subject(args.subject)
    frequencies = sorted(set(dataset.frequencies_hz))
    data = prepare_eeg(crop_window(dataset, args.window), select_channels(dataset))
    trial = data[args.trial]
    decoder = SSVEPDecoder(frequencies, FBCCAConfig(window_seconds=args.window))
    selection = decoder.predict(trial)
    scores = decoder.score(trial)
    print("SSVEP TARGET SCORES")
    for frequency, score in zip(frequencies, scores):
        print(f"{frequency:5.1f} Hz  {score:.6f}")
    print(f"Prediction: {selection.target_id} ({selection.frequency_hz:.1f} Hz)")
    print(f"Score margin: {selection.score_margin:.6f}")
    print(f"Ground truth: {dataset.frequencies_hz[args.trial]:.1f} Hz")
    if args.plot:
        save_score_plot(frequencies, scores, selection.frequency_hz, args.plot)
        print(f"Plot saved: {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
