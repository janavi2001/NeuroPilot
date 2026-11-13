"""Exact Step 1 preprocessing used by the LEAF BCIC-IV-2a builder."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import scipy.io

from .decoder import LEAF_CHANNELS


EVENT_LABELS = {769: "LEFT", 770: "RIGHT"}
MAT_LABELS = {0: "LEFT", 1: "RIGHT", 2: "FOOT", 3: "TONGUE"}


def preprocess_epochs(epochs: mne.Epochs) -> np.ndarray:
    """Convert 22-channel 250-Hz, 4-second epochs into ``(N, 65, 800)`` LEAF input."""

    leaf_root = Path(__file__).resolve().parents[1] / "third_party" / "LEAF"
    import sys
    if str(leaf_root) not in sys.path:
        sys.path.insert(0, str(leaf_root))
    from leaf_datasets.shared import pipeline

    data = epochs.get_data(picks=list(LEAF_CHANNELS)).astype(np.float32)
    if data.shape[1:] != (22, 1000):
        raise ValueError(f"Expected raw epochs shaped (N, 22, 1000), got {data.shape}")
    data = epochs.copy().filter(0.3, 40.0, picks=list(LEAF_CHANNELS), verbose=False).resample(200, verbose=False).get_data(picks=list(LEAF_CHANNELS)).astype(np.float32)
    return pipeline(data, list(LEAF_CHANNELS)).astype(np.float32)


def load_real_trials(path: str | Path, limit: int | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load labeled LEFT/RIGHT trials from a real BCI-IV-2a file."""

    path = Path(path)
    if path.suffix.lower() == ".mat":
        return _load_mat_trials(path, limit)

    raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    wanted = {str(code): code for code in EVENT_LABELS}
    selected = {name: code for name, code in event_id.items() if name in wanted}
    if not selected:
        selected = {str(code): code for code in EVENT_LABELS if code in events[:, 2]}
    if not selected:
        raise ValueError(f"No class events 769/770 found; event mapping was {event_id}")
    epochs = mne.Epochs(raw, events, event_id=selected, tmin=0.0, tmax=4.0 - 1.0 / raw.info["sfreq"], baseline=None, preload=True, picks=list(LEAF_CHANNELS), verbose=False)
    labels = np.array([EVENT_LABELS[int(code)] for code in epochs.events[:, 2]])
    if limit is not None:
        epochs = epochs[:limit]
        labels = labels[:limit]
    processed = preprocess_epochs(epochs)
    metadata = {
        "path": str(Path(path).resolve()),
        "channels": 22,
        "sampling_rate_hz": float(raw.info["sfreq"]),
        "raw_trial_samples": int(round(4.0 * raw.info["sfreq"])),
        "processed_shape": list(processed.shape),
        "filter_hz": [0.3, 40.0],
        "resampled_rate_hz": 200,
        "normalization": "LEAF percentile clipping and global robust median/IQR scaling, then 65-channel montage mapping",
        "label_mapping": EVENT_LABELS,
    }
    return processed, labels, metadata


def _load_mat_trials(path: Path, limit: int | None) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Use the released LEAF BCIC-IV-2a MAT preprocessing, then keep two classes."""

    leaf_root = Path(__file__).resolve().parents[1] / "third_party" / "LEAF"
    import sys
    if str(leaf_root) not in sys.path:
        sys.path.insert(0, str(leaf_root))
    from leaf_datasets.MI_BCIC_IV2a import proc_one_mat
    from leaf_datasets.shared import Param, pipeline

    source = scipy.io.loadmat(path)
    raw_trials, raw_labels = proc_one_mat(source)
    info = mne.create_info(list(LEAF_CHANNELS), sfreq=250, ch_types="eeg", verbose=False)
    epochs = mne.EpochsArray(raw_trials, info, tmin=0, verbose=False)
    epochs.filter(l_freq=Param.MI_lp, h_freq=Param.MI_hp, verbose=False)
    epochs.resample(Param.resample, npad="auto", verbose=False)
    processed = epochs.get_data().astype(np.float32)[:, :, 400:1200]
    processed = pipeline(processed, list(LEAF_CHANNELS)).astype(np.float32)

    keep = np.isin(raw_labels, [0, 1])
    processed = processed[keep]
    labels = np.array([MAT_LABELS[int(label)] for label in raw_labels[keep]])
    if limit is not None:
        processed = processed[:limit]
        labels = labels[:limit]
    metadata = {
        "path": str(path.resolve()),
        "channels": 22,
        "sampling_rate_hz": 250,
        "raw_trial_samples": 1500,
        "processed_shape": list(processed.shape),
        "filter_hz": [float(Param.MI_lp), float(Param.MI_hp)],
        "resampled_rate_hz": int(Param.resample),
        "normalization": "LEAF percentile clipping and global robust median/IQR scaling, then 65-channel montage mapping",
        "label_mapping": MAT_LABELS,
        "selected_labels": ["LEFT", "RIGHT"],
    }
    return processed, labels, metadata
