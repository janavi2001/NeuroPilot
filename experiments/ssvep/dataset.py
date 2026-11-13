"""BETA SSVEP loading and metadata normalization through MOABB."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SSVEPDataset:
    data: np.ndarray  # trials, channels, samples
    frequencies_hz: np.ndarray  # trials
    phases_rad: np.ndarray  # trials, when available
    channel_names: tuple[str, ...]
    sampling_rate_hz: float
    subject: int
    stimulus_onset_seconds: float
    stimulus_duration_seconds: float
    metadata: dict[str, Any]


def load_beta_subject(subject: int = 1, path: str | Path | None = None) -> SSVEPDataset:
    """Load one real BETA subject using MOABB's official dataset adapter."""

    if subject < 1 or subject > 70:
        raise ValueError("BETA subject must be in the range 1..70")
    from moabb.datasets import Liu2020BETA

    dataset = Liu2020BETA()
    raw_by_session = dataset.get_data(subjects=[subject])
    raw = raw_by_session[subject]["0"]["0"]
    import mne

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    epochs = _epochs_from_raw(raw, events, event_id)
    return _normalize_epochs(epochs, subject, raw)


def _epochs_from_raw(raw: Any, events: np.ndarray, event_id: dict[str, int]) -> Any:
    import mne

    return mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0.0,
        tmax=2.996,
        baseline=None,
        preload=True,
        verbose=False,
    )


def _normalize_epochs(epochs: Any, subject: int, raw: Any) -> SSVEPDataset:
    data = epochs.get_data(copy=True).astype(np.float64)
    inverse_event_id = {value: key for key, value in epochs.event_id.items()}
    labels = [inverse_event_id[int(code)] for code in epochs.events[:, 2]]
    frequencies = np.array([_frequency_from_label(label) for label in labels], dtype=float)
    phases = np.array([_phase_from_label(label) for label in labels], dtype=float)
    duration = 2.0 if subject <= 15 else 3.0
    return SSVEPDataset(
        data=data,
        frequencies_hz=frequencies,
        phases_rad=phases,
        channel_names=tuple(raw.ch_names),
        sampling_rate_hz=float(raw.info["sfreq"]),
        subject=subject,
        stimulus_onset_seconds=0.5,
        stimulus_duration_seconds=duration,
        metadata={"dataset": "Liu2020BETA", "blocks": 4, "targets": 40},
    )


def _frequency_from_label(label: str) -> float:
    # MOABB event labels are frequencies such as "8.0" or "10.2".
    try:
        return float(label)
    except ValueError as exc:
        raise ValueError(f"Could not parse BETA frequency label {label!r}") from exc


def _phase_from_label(label: str) -> float:
    # BETA's class labels identify frequency; phase is not needed for the
    # phase-invariant CCA baseline. Keep the field explicit for future use.
    return 0.0


def select_channels(dataset: SSVEPDataset, requested: tuple[str, ...] | None = None) -> tuple[int, ...]:
    names = {name.upper(): index for index, name in enumerate(dataset.channel_names)}
    channels = requested or ("O1", "Oz", "O2", "PO3", "PO4", "PO7", "PO8", "POz")
    missing = [name for name in channels if name.upper() not in names]
    if missing:
        raise ValueError(f"Requested SSVEP channels are missing: {missing}")
    return tuple(names[name.upper()] for name in channels)


def crop_window(dataset: SSVEPDataset, seconds: float) -> np.ndarray:
    if seconds <= 0 or seconds > dataset.stimulus_duration_seconds:
        raise ValueError(
            f"Window must be in (0, {dataset.stimulus_duration_seconds}] seconds for subject {dataset.subject}"
        )
    start = int(round(dataset.stimulus_onset_seconds * dataset.sampling_rate_hz))
    length = int(round(seconds * dataset.sampling_rate_hz))
    return dataset.data[:, :, start : start + length]
