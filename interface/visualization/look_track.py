"""Precomputed per-frame decoder views of one recorded SSVEP look."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from bci.ssvep_decoder import SSVEPDecoder

TOPOMAP_CHECKPOINTS = (0.5, 1.0, 1.5, 2.0)
MIN_GROWING_WINDOW = 0.25


@dataclass
class LookTrack:
    """One real recorded trial, decoded on windows that grow with playback time.

    ``scores_at(t)`` is what the CCA decoder outputs on the first ``t`` seconds,
    so the score race in the video is computed, not animated. The final decision
    uses the full configured window and threshold, exactly as in Step 4b.
    """

    intended: str
    trial_number: int
    eeg: np.ndarray  # decoder channels x samples
    eeg_all: np.ndarray  # all EEG channels with positions x samples
    sampling_rate_hz: float
    targets: dict[str, float]
    final: dict[str, Any]
    frame_times: np.ndarray
    frame_scores: np.ndarray  # frames x targets (NaN before MIN_GROWING_WINDOW)
    topomap_values: dict[float, tuple[str, np.ndarray]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        intended: str,
        trial_number: int,
        eeg: np.ndarray,
        eeg_all: np.ndarray,
        decoder: SSVEPDecoder,
        fps: int,
    ) -> "LookTrack":
        fs = decoder.sampling_rate_hz
        window = decoder.window_seconds
        frame_times = np.arange(1, int(round(window * fps)) + 1) / fps
        names = list(decoder.targets)
        scores = np.full((len(frame_times), len(names)), np.nan)
        for index, t in enumerate(frame_times):
            if t < MIN_GROWING_WINDOW:
                continue
            partial = SSVEPDecoder(decoder.targets, decoder.n_channels, fs, round(t * fs) / fs, decoder.harmonics)
            result = partial.predict(np.ascontiguousarray(eeg[:, : partial.samples]))
            scores[index] = [result["scores"][name] for name in names]
        final = decoder.predict(eeg)
        track = cls(intended, trial_number, eeg, eeg_all, fs, dict(decoder.targets), final, frame_times, scores)
        for checkpoint in TOPOMAP_CHECKPOINTS:
            samples = int(round(checkpoint * fs))
            row = scores[min(len(frame_times) - 1, int(round(checkpoint * fps)) - 1)]
            leader = names[int(np.nanargmax(row))]
            track.topomap_values[checkpoint] = (leader, ssvep_amplitude(eeg_all[:, :samples], decoder.targets[leader], fs))
        return track

    def scores_at(self, t: float) -> np.ndarray:
        index = int(np.clip(np.searchsorted(self.frame_times, t + 1e-9) - 1, 0, len(self.frame_times) - 1))
        return self.frame_scores[index]

    def topomap_at(self, t: float) -> tuple[float, str, np.ndarray] | None:
        reached = [c for c in TOPOMAP_CHECKPOINTS if c <= t + 1e-9]
        if not reached:
            return None
        leader, values = self.topomap_values[reached[-1]]
        return reached[-1], leader, values


def ssvep_amplitude(eeg: np.ndarray, frequency: float, fs: float, harmonics: int = 2) -> np.ndarray:
    """Per-channel amplitude at ``frequency`` and its harmonics (sine/cosine projection)."""
    time = np.arange(eeg.shape[1]) / fs
    centered = eeg - eeg.mean(axis=1, keepdims=True)
    amplitude = np.zeros(eeg.shape[0])
    for harmonic in range(1, harmonics + 1):
        phase = 2 * np.pi * harmonic * frequency * time
        amplitude += np.hypot(centered @ np.sin(phase), centered @ np.cos(phase)) * 2 / eeg.shape[1]
    return amplitude


def spectrum(eeg: np.ndarray, fs: float, n_fft: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed magnitude spectrum of the channel mean."""
    signal = eeg.mean(axis=0)
    signal = (signal - signal.mean()) * np.hanning(len(signal))
    magnitude = np.abs(np.fft.rfft(signal, n=n_fft))
    return np.fft.rfftfreq(n_fft, 1 / fs), magnitude


def tracks_for_sequence(truths: Sequence[str]) -> dict[str, list[int]]:
    """Recorded trial indices grouped by ground-truth target, in recording order."""
    grouped: dict[str, list[int]] = {}
    for index, truth in enumerate(truths):
        grouped.setdefault(truth, []).append(index)
    return grouped
