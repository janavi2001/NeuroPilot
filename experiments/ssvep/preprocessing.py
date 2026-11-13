"""Small, explicit preprocessing helpers for SSVEP decoding."""

from __future__ import annotations

import numpy as np
from scipy.signal import detrend


def prepare_eeg(data: np.ndarray, channel_indices: tuple[int, ...]) -> np.ndarray:
    """Select posterior channels and remove per-channel DC/trend."""

    if data.ndim != 3:
        raise ValueError(f"Expected EEG shape (trials, channels, samples), got {data.shape}")
    selected = data[:, channel_indices, :].astype(np.float64, copy=True)
    return detrend(selected, axis=-1, type="constant")
