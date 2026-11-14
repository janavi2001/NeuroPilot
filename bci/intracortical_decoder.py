"""Reach-intention decoder for intracortical spiking: which direction will the subject reach?

A shrinkage linear discriminant on square-root spike counts from a window that
ends before (or at) movement onset. This is a small fitted adapter: no
pretrained decoder exists for the NLB MC_Maze data, so it is fitted on the
official train split and evaluated on the held-out val split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class DecoderWindow:
    start: float  # seconds relative to movement onset
    stop: float
    bin_seconds: float = 0.1


class ReachIntentDecoder:
    def __init__(self, classes: Sequence[str], window: DecoderWindow) -> None:
        if window.stop <= window.start or window.bin_seconds <= 0:
            raise ValueError(f"Invalid decoder window {window}")
        self.classes = tuple(classes)
        self.window = window
        self._model = None
        self._mean = self._scale = None

    def features(self, trials, indices: Sequence[int]) -> np.ndarray:
        w = self.window
        return np.stack([np.sqrt(trials.spike_counts(i, w.start, w.stop, w.bin_seconds)).ravel() for i in indices])

    def fit(self, trials, indices: Sequence[int]) -> "ReachIntentDecoder":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        x = self.features(trials, indices)
        y = np.asarray(trials.labels)[list(indices)]
        missing = set(self.classes) - set(y)
        if missing:
            raise ValueError(f"Training data has no trials for {sorted(missing)}")
        self._mean = x.mean(axis=0)
        self._scale = x.std(axis=0) + 1e-6
        self._model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self._model.fit((x - self._mean) / self._scale, y)
        return self

    def predict_features(self, x: np.ndarray) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("Decoder is not fitted")
        if x.shape != self._mean.shape:
            raise ValueError(f"Expected {self._mean.shape[0]} features, got {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("features contain non-finite values")
        probabilities = self._model.predict_proba(((x - self._mean) / self._scale)[None])[0]
        by_class = {str(c): float(p) for c, p in zip(self._model.classes_, probabilities)}
        ranked = sorted(by_class, key=by_class.__getitem__, reverse=True)
        return {
            "intent": ranked[0],
            "score": by_class[ranked[0]],
            "margin": by_class[ranked[0]] - by_class[ranked[1]],
            "scores": {name: by_class.get(name, 0.0) for name in self.classes},
        }

    def predict_trial(self, trials, index: int) -> dict[str, Any]:
        return self.predict_features(self.features(trials, [index])[0])
