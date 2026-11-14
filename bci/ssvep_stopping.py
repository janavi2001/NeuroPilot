"""Dynamic stopping for SSVEP: decide as soon as the CCA margin is large enough."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .ssvep_decoder import UNSURE, SSVEPDecoder


@dataclass(frozen=True)
class StoppingRule:
    """Stop at the first checkpoint t ≥ ``min_seconds`` where margin ≥ base_margin · (T / t)^exponent.

    T is the full window. Short windows give noisy margins, so the required margin
    is larger early and relaxes towards ``base_margin`` at T. If no checkpoint
    qualifies, the full window is decoded with the usual UNSURE threshold,
    exactly like the fixed-window decoder.
    """

    min_seconds: float
    base_margin: float
    exponent: float = 0.0
    step_seconds: float = 0.1

    def required_margin(self, seconds: float, window_seconds: float) -> float:
        return self.base_margin * (window_seconds / seconds) ** self.exponent

    @classmethod
    def from_yaml(cls, path) -> "StoppingRule":
        import yaml
        from pathlib import Path

        section = yaml.safe_load(Path(path).read_text()).get("early_stop")
        if section is None:
            raise ValueError(f"No early_stop section in {path}")
        return cls(float(section["min_seconds"]), float(section["base_margin"]), float(section["exponent"]))


class EarlyStoppingDecoder:
    def __init__(self, targets: Mapping[str, float], n_channels: int, rule: StoppingRule,
                 sampling_rate_hz: float = 250.0, window_seconds: float = 2.0, harmonics: int = 3,
                 margin_threshold: float = 0.0) -> None:
        if not 0 < rule.min_seconds <= window_seconds or rule.base_margin < 0 or rule.exponent < 0 or rule.step_seconds <= 0:
            raise ValueError(f"Invalid stopping rule {rule} for a {window_seconds} s window")
        self.rule = rule
        self.full = SSVEPDecoder(targets, n_channels, sampling_rate_hz, window_seconds, harmonics, margin_threshold)
        count = int(round((window_seconds - rule.min_seconds) / rule.step_seconds))
        self.checkpoints = [round(rule.min_seconds + i * rule.step_seconds, 6) for i in range(count)
                            if rule.min_seconds + i * rule.step_seconds < window_seconds - 1e-9]
        self._partial = [SSVEPDecoder(targets, n_channels, sampling_rate_hz, t, harmonics) for t in self.checkpoints]

    def predict(self, eeg_window: np.ndarray) -> dict[str, Any]:
        """Replay ``eeg_window`` as a stream; return the decision and when it was made."""
        self.full._validate(eeg_window)
        for seconds, decoder in zip(self.checkpoints, self._partial):
            result = decoder.predict(np.ascontiguousarray(eeg_window[:, : decoder.samples]))
            if result["margin"] >= self.rule.required_margin(seconds, self.full.window_seconds):
                return {**result, "decision_seconds": seconds, "early": True}
        result = self.full.predict(eeg_window)
        return {**result, "decision_seconds": self.full.window_seconds, "early": False}


def margins_over_time(eeg: np.ndarray, targets: Mapping[str, float], fs: float, times: list[float],
                      harmonics: int = 3) -> tuple[np.ndarray, list[str]]:
    """Best target and margin of the CCA decoder at each window length (for evaluation)."""
    names = list(targets)
    best, margins = [], []
    for seconds in times:
        decoder = SSVEPDecoder(targets, eeg.shape[0], fs, seconds, harmonics)
        result = decoder.predict(np.ascontiguousarray(eeg[:, : decoder.samples]))
        best.append(names.index(result["best_target"]))
        margins.append(result["margin"])
    return np.array([best, margins]), names


__all__ = ["EarlyStoppingDecoder", "StoppingRule", "UNSURE", "margins_over_time"]
