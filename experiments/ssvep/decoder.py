"""Training-free CCA and filter-bank CCA SSVEP decoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.cross_decomposition import CCA


@dataclass(frozen=True)
class SSVEPSelection:
    target_id: str
    frequency_hz: float
    score: float
    score_margin: float
    window_seconds: float


@dataclass(frozen=True)
class CCAConfig:
    sampling_rate_hz: float = 250.0
    harmonics: int = 3
    phase_rad: float = 0.0
    window_seconds: float = 2.0


@dataclass(frozen=True)
class FBCCAConfig(CCAConfig):
    bands_hz: tuple[tuple[float, float], ...] = (
        (6.0, 90.0),
        (8.0, 90.0),
        (10.0, 90.0),
        (12.0, 90.0),
        (14.0, 90.0),
    )
    filter_order: int = 4
    weight_exponent: float = 1.25
    weight_offset: float = 0.25


class SSVEPDecoder:
    def __init__(self, frequencies_hz: Iterable[float], config: CCAConfig | FBCCAConfig | None = None):
        self.frequencies_hz = tuple(float(frequency) for frequency in frequencies_hz)
        if not self.frequencies_hz:
            raise ValueError("At least one target frequency is required")
        self.config = config or CCAConfig()

    def predict(self, eeg: np.ndarray) -> SSVEPSelection:
        scores = self.score(eeg)
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        second = float(scores[order[1]]) if len(order) > 1 else 0.0
        return SSVEPSelection(
            target_id=f"target_{best + 1:02d}",
            frequency_hz=self.frequencies_hz[best],
            score=float(scores[best]),
            score_margin=float(scores[best] - second),
            window_seconds=eeg.shape[-1] / self.config.sampling_rate_hz,
        )

    def score(self, eeg: np.ndarray) -> np.ndarray:
        if eeg.ndim != 2 or eeg.shape[1] < 8:
            raise ValueError(f"Expected EEG shape (channels, samples), got {eeg.shape}")
        if not np.isfinite(eeg).all():
            raise ValueError("EEG contains non-finite values")
        references = [_reference_signals(frequency, eeg.shape[1], self.config) for frequency in self.frequencies_hz]
        if isinstance(self.config, FBCCAConfig):
            return self._fbcca_scores(eeg, references, self.config)
        return np.array([_cca_score(eeg, reference) for reference in references])

    @staticmethod
    def _fbcca_scores(eeg, references, config: FBCCAConfig) -> np.ndarray:
        scores = np.zeros(len(references), dtype=float)
        for band_index, (low, high) in enumerate(config.bands_hz):
            nyquist = config.sampling_rate_hz / 2.0
            if high >= nyquist:
                raise ValueError(f"Filter high cutoff must be below Nyquist ({nyquist})")
            sos = butter(config.filter_order, (low, high), btype="bandpass", fs=config.sampling_rate_hz, output="sos")
            filtered = sosfiltfilt(sos, eeg, axis=-1)
            weight = (band_index + 1) ** (-config.weight_exponent) + config.weight_offset
            scores += weight * np.array([_cca_score(filtered, reference) ** 2 for reference in references])
        return scores


def _reference_signals(frequency: float, samples: int, config: CCAConfig) -> np.ndarray:
    time = np.arange(samples, dtype=float) / config.sampling_rate_hz
    columns = []
    for harmonic in range(1, config.harmonics + 1):
        phase = 2 * np.pi * harmonic * frequency * time + harmonic * config.phase_rad
        columns.extend((np.sin(phase), np.cos(phase)))
    return np.column_stack(columns)


def _cca_score(eeg: np.ndarray, reference: np.ndarray) -> float:
    cca = CCA(n_components=1, scale=False, max_iter=1000)
    cca.fit(eeg.T, reference)
    eeg_projection, reference_projection = cca.transform(eeg.T, reference)
    if np.std(eeg_projection) == 0 or np.std(reference_projection) == 0:
        return 0.0
    return float(abs(np.corrcoef(eeg_projection[:, 0], reference_projection[:, 0])[0, 1]))
