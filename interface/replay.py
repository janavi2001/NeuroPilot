"""Recorded EEG replay for Step 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Decoder(Protocol):
    def predict(self, eeg_window: np.ndarray) -> dict[str, object]: ...


@dataclass(frozen=True)
class ReplayEvent:
    trial_number: int
    eeg_window: np.ndarray
    ground_truth: str
    prediction: dict[str, object]


class RecordedEEGReplay:
    """Turn preprocessed recorded trials into decoder events, one trial at a time."""

    def __init__(self, eeg: np.ndarray, labels: np.ndarray, decoder: Decoder) -> None:
        if eeg.ndim != 3 or eeg.shape[0] == 0:
            raise ValueError(f"Expected non-empty EEG array (N, C, T), got {eeg.shape}")
        if len(labels) != len(eeg):
            raise ValueError("EEG trial count and label count must match")
        self.eeg = eeg
        self.labels = labels
        self.decoder = decoder

    def __iter__(self):
        for index, (window, label) in enumerate(zip(self.eeg, self.labels), start=1):
            yield ReplayEvent(index, window, str(label), self.decoder.predict(window))
