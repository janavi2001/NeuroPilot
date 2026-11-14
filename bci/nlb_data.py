"""Neural Latents Benchmark MC_Maze trials (macaque M1/PMd spiking, delayed reaching).

Reads the DANDI NWB file directly with h5py. Reach targets are grouped into four
directions that map onto the robot table layout used by the SSVEP frontend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

DEFAULT_NWB = Path("data/raw/nlb_mc_maze_large/sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb")
DANDI_URL = "https://api.dandiarchive.org/api/assets/e67b57b2-e9ad-4d95-b9e3-1262997360dc/download/"

# Reach direction (degrees, maze screen coordinates) -> selection target.
DIRECTION_TARGETS = ("APPLE", "CUP", "PHONE", "CANCEL")
# Six classes for the five-object table, chosen on the train split by 5-fold CV
# (84.7% balanced, versus 76.9% for splitting the 30/39 degree upper-right pair).
# Ordered by angle so the table's right-to-left order follows the reach direction.
SIX_DIRECTION_TARGETS = ("BANANA", "APPLE", "CUP", "BOTTLE", "PHONE", "CANCEL")
SCHEME_TARGETS = {"four": DIRECTION_TARGETS, "six": SIX_DIRECTION_TARGETS}


def six_direction_class(target_xy: np.ndarray) -> str:
    """-174 deg -> BANANA, 173 -> APPLE, 144 -> CUP, 82 -> BOTTLE, 30-39 -> PHONE, -33 -> CANCEL."""
    angle = float(np.degrees(np.arctan2(target_xy[1], target_xy[0])))
    for name, low, high in (("BANANA", -180, -160), ("APPLE", 160, 180.01), ("CUP", 120, 160), ("BOTTLE", 60, 120),
                            ("PHONE", 0, 60), ("CANCEL", -60, 0)):
        if low <= angle < high:
            return name
    raise ValueError(f"Reach target {target_xy} at {angle:.0f} degrees has no direction class")


def direction_class(target_xy: np.ndarray, scheme: str = "four") -> str:
    """left/upper-left -> APPLE, up -> CUP, upper-right -> PHONE, lower-right -> CANCEL."""
    if scheme == "six":
        return six_direction_class(target_xy)
    if scheme != "four":
        raise ValueError(f"Unknown direction scheme {scheme!r}")
    angle = float(np.degrees(np.arctan2(target_xy[1], target_xy[0])))
    if angle >= 120 or angle <= -120:
        return "APPLE"
    if 60 <= angle < 120:
        return "CUP"
    if 0 <= angle < 60:
        return "PHONE"
    if -60 <= angle < 0:
        return "CANCEL"
    raise ValueError(f"Reach target {target_xy} at {angle:.0f} degrees has no direction class")


@dataclass(frozen=True)
class ReachTrials:
    spike_times: list[np.ndarray]  # per unit, seconds
    move_onset: np.ndarray  # per trial, seconds
    target_on: np.ndarray
    go_cue: np.ndarray
    target_xy: np.ndarray  # trials x 2
    labels: np.ndarray  # direction class per trial
    split: np.ndarray  # "train" or "val" (official NLB split)
    hand_pos: np.ndarray  # samples x 2
    hand_time: np.ndarray  # samples
    cursor_pos: np.ndarray  # samples x 2, screen coordinates shared with target_xy

    @property
    def n_units(self) -> int:
        return len(self.spike_times)

    def spike_counts(self, trial: int, start: float, stop: float, bin_seconds: float) -> np.ndarray:
        """Units x bins spike counts in [onset+start, onset+stop) relative to movement onset."""
        onset = self.move_onset[trial]
        edges = onset + np.arange(start, stop + 1e-9, bin_seconds)
        return np.stack([np.histogram(times[(times >= edges[0]) & (times < edges[-1])], edges)[0]
                         for times in self.spike_times]).astype(float)

    def cursor_at(self, trial: int, seconds: float) -> np.ndarray:
        """Cursor position at ``seconds`` relative to movement onset."""
        index = int(np.searchsorted(self.hand_time, self.move_onset[trial] + seconds))
        return self.cursor_pos[min(index, len(self.cursor_pos) - 1)]


def load_mc_maze(path: str | Path = DEFAULT_NWB, scheme: str = "four") -> ReachTrials:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; download it from {DANDI_URL}")
    with h5py.File(path, "r") as f:
        units = f["units"]
        index = units["spike_times_index"][:]
        flat = units["spike_times"][:]
        spike_times = np.split(flat, index[:-1])
        trials = f["intervals/trials"]
        positions = trials["target_pos"][:]
        ends = trials["target_pos_index"][:]
        starts = np.concatenate([[0], ends[:-1]])
        active = trials["active_target"][:]
        target_xy = np.array([positions[s + a] for s, a in zip(starts, active)], dtype=float)
        if not trials["success"][:].all():
            raise ValueError("Expected only successful trials in the NLB train file")
        behavior = f["processing/behavior/hand_pos"]
        return ReachTrials(
            spike_times=[np.sort(t) for t in spike_times],
            move_onset=trials["move_onset_time"][:],
            target_on=trials["target_on_time"][:],
            go_cue=trials["go_cue_time"][:],
            target_xy=target_xy,
            labels=np.array([direction_class(xy, scheme) for xy in target_xy]),
            split=np.array([s.decode() if isinstance(s, bytes) else str(s) for s in trials["split"][:]]),
            hand_pos=behavior["data"][:],
            hand_time=behavior["timestamps"][:],
            cursor_pos=f["processing/behavior/cursor_pos/data"][:],
        )
