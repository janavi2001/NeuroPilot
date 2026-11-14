"""Training-free CCA decoder for multi-target SSVEP selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


UNSURE = "UNSURE"


@dataclass(frozen=True)
class SSVEPPrediction:
    """A decoder result with raw canonical correlations, not probabilities."""

    intent: str
    accepted: bool
    best_target: str
    frequency_hz: float
    score: float
    margin: float
    scores: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SSVEPDecoder:
    """Score each target frequency by exact CCA against sine/cosine references.

    Input windows are ``(channels, samples)`` arrays already cropped to the
    stimulation window. Nothing is trained or fitted to the EEG. When the gap
    between the best and second-best correlation is below ``margin_threshold``
    the decoder returns ``UNSURE`` instead of guessing.
    """

    def __init__(
        self,
        targets: Mapping[str, float],
        n_channels: int,
        sampling_rate_hz: float = 250.0,
        window_seconds: float = 2.0,
        harmonics: int = 3,
        margin_threshold: float = 0.0,
    ) -> None:
        if len(targets) < 2:
            raise ValueError("At least two SSVEP targets are required")
        frequencies = [float(frequency) for frequency in targets.values()]
        if len(set(frequencies)) != len(frequencies) or min(frequencies) <= 0:
            raise ValueError(f"Target frequencies must be distinct and positive: {dict(targets)}")
        if UNSURE in targets:
            raise ValueError(f"{UNSURE!r} is reserved and cannot be a target name")
        if harmonics < 1 or margin_threshold < 0 or n_channels < 1:
            raise ValueError("harmonics and n_channels must be positive; margin_threshold must be non-negative")
        if max(frequencies) * harmonics >= sampling_rate_hz / 2:
            raise ValueError("Highest reference harmonic must stay below the Nyquist frequency")

        self.targets = {str(name): float(frequency) for name, frequency in targets.items()}
        self.n_channels = n_channels
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.window_seconds = float(window_seconds)
        self.samples = int(round(window_seconds * sampling_rate_hz))
        self.harmonics = harmonics
        self.margin_threshold = float(margin_threshold)
        time = np.arange(self.samples, dtype=float) / self.sampling_rate_hz
        self._reference_bases = {name: _orthonormal_basis(_references(frequency, time, harmonics)) for name, frequency in self.targets.items()}

    def predict(self, eeg_window: np.ndarray) -> dict[str, Any]:
        """Return the attended target (or ``UNSURE``), its score, margin, and all scores."""

        self._validate(eeg_window)
        eeg_basis = _orthonormal_basis(eeg_window.T.astype(float))
        scores = {name: _canonical_correlation(eeg_basis, reference) for name, reference in self._reference_bases.items()}
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        best = ranked[0]
        margin = scores[best] - scores[ranked[1]]
        accepted = margin >= self.margin_threshold
        return SSVEPPrediction(
            intent=best if accepted else UNSURE,
            accepted=accepted,
            best_target=best,
            frequency_hz=self.targets[best],
            score=scores[best],
            margin=margin,
            scores=scores,
        ).as_dict()

    def _validate(self, eeg_window: np.ndarray) -> None:
        if not isinstance(eeg_window, np.ndarray):
            raise TypeError(f"eeg_window must be a numpy array, got {type(eeg_window).__name__}")
        expected = (self.n_channels, self.samples)
        if eeg_window.shape != expected:
            raise ValueError(f"Expected SSVEP input shape {expected}, got {eeg_window.shape}")
        if eeg_window.dtype.kind != "f":
            raise TypeError(f"eeg_window must contain floating-point values, got {eeg_window.dtype}")
        if not np.isfinite(eeg_window).all():
            raise ValueError("eeg_window contains non-finite values")
        if (eeg_window.std(axis=1) == 0).any():
            raise ValueError("eeg_window contains a flat channel")


def _references(frequency: float, time: np.ndarray, harmonics: int) -> np.ndarray:
    columns = []
    for harmonic in range(1, harmonics + 1):
        phase = 2 * np.pi * harmonic * frequency * time
        columns.extend((np.sin(phase), np.cos(phase)))
    return np.column_stack(columns)


def _orthonormal_basis(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0)
    basis, _ = np.linalg.qr(centered)
    return basis


def _canonical_correlation(x_basis: np.ndarray, y_basis: np.ndarray) -> float:
    # The largest singular value of Qx^T Qy is the first canonical correlation.
    return float(min(1.0, np.linalg.svd(x_basis.T @ y_basis, compute_uv=False)[0]))
