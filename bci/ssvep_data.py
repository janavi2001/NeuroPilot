"""Real BETA SSVEP trials loaded through MOABB for replay."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


BETA_ONSET_SECONDS = 0.5


def beta_stimulus_seconds(subject: int) -> float:
    """BETA S1–S15 flicker for 2 s after the cue; S16–S70 flicker for 3 s."""

    return 2.0 if subject <= 15 else 3.0


def load_beta_trials(
    subject: int,
    channels: Sequence[str],
    window_seconds: float,
    frequencies_hz: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return ``(trials, channels, samples)`` EEG starting at stimulus onset.

    Trials keep recording order. When ``frequencies_hz`` is given, only trials
    whose real stimulus frequency is in that list are returned.
    """

    if not 1 <= subject <= 70:
        raise ValueError("BETA subject must be in the range 1..70")
    if not 0 < window_seconds <= beta_stimulus_seconds(subject):
        raise ValueError(f"Window must be in (0, {beta_stimulus_seconds(subject)}] seconds for BETA subject {subject}")

    import mne
    from moabb.datasets import Liu2020BETA

    raw = Liu2020BETA().get_data(subjects=[subject])[subject]["0"]["0"]
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    sfreq = float(raw.info["sfreq"])
    start = int(round(BETA_ONSET_SECONDS * sfreq))
    samples = int(round(window_seconds * sfreq))
    epochs = mne.Epochs(
        raw, events, event_id=event_id, tmin=0.0, tmax=(start + samples - 1) / sfreq,
        baseline=None, preload=True, picks=list(channels), verbose=False,
    )
    code_to_frequency = {code: float(label) for label, code in event_id.items()}
    labels = np.array([code_to_frequency[int(code)] for code in epochs.events[:, 2]])
    data = epochs.get_data(copy=True)[:, :, start : start + samples]
    if frequencies_hz is not None:
        keep = np.isin(labels, np.asarray(frequencies_hz, dtype=float))
        data, labels = data[keep], labels[keep]
    metadata = {
        "dataset": "Liu2020BETA (MOABB)",
        "subject": subject,
        "channels": list(epochs.ch_names),
        "sampling_rate_hz": sfreq,
        "window_seconds": window_seconds,
        "window_start_seconds_after_trial_start": BETA_ONSET_SECONDS,
        "trials": int(len(labels)),
    }
    return data.astype(np.float64), labels, metadata
