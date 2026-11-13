"""Proof-of-concept score visualization for recorded SSVEP replay."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_score_plot(frequencies, scores, prediction_frequency, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#2ea043" if frequency == prediction_frequency else "#58a6ff" for frequency in frequencies]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar([f"{frequency:.1f} Hz" for frequency in frequencies], scores, color=colors)
    axis.set_title("Recorded SSVEP target scores")
    axis.set_ylabel("CCA / FBCCA score (not probability)")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
