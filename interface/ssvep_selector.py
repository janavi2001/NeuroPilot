"""Direct object selection from SSVEP decoder events."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bci.ssvep_decoder import UNSURE
from .selector import SelectionState


DEFAULT_CONFIG = Path(__file__).with_name("ssvep_targets.yaml")


@dataclass(frozen=True)
class SSVEPConfig:
    targets: dict[str, float]
    cancel_target: str
    channels: tuple[str, ...]
    window_seconds: float
    harmonics: int
    margin_threshold: float

    @property
    def objects(self) -> tuple[str, ...]:
        return tuple(name for name in self.targets if name != self.cancel_target)


def load_ssvep_config(path: str | Path = DEFAULT_CONFIG) -> SSVEPConfig:
    raw = yaml.safe_load(Path(path).read_text())
    targets = {str(name).upper(): float(frequency) for name, frequency in raw["targets"].items()}
    cancel = str(raw["cancel_target"]).upper()
    if cancel not in targets:
        raise ValueError(f"cancel_target {cancel!r} must be one of the targets {list(targets)}")
    return SSVEPConfig(
        targets=targets,
        cancel_target=cancel,
        channels=tuple(raw["channels"]),
        window_seconds=float(raw["window_seconds"]),
        harmonics=int(raw["harmonics"]),
        margin_threshold=float(raw["margin_threshold"]),
    )


class SSVEPSelector:
    """Looking at an object selects it; CANCEL clears; UNSURE changes nothing."""

    def __init__(self, objects: tuple[str, ...], cancel_target: str = "CANCEL") -> None:
        if not objects or cancel_target in objects:
            raise ValueError("objects must be non-empty and must not include the cancel target")
        self._objects = tuple(name.upper() for name in objects)
        self._cancel = cancel_target.upper()
        self._index = 0
        self._selected: str | None = None
        self._history: list[str] = []

    @property
    def state(self) -> SelectionState:
        return SelectionState(self._objects, self._index, self._selected, tuple(self._history))

    def apply(self, prediction: dict[str, Any]) -> SelectionState:
        intent = str(prediction["intent"]).upper()
        if intent == UNSURE:
            self._history.append("UNSURE -> no change")
        elif intent == self._cancel:
            self._selected = None
            self._history.append("CANCEL -> selection cleared")
        elif intent in self._objects:
            self._index = self._objects.index(intent)
            self._selected = intent
            self._history.append(f"LOOK {intent} -> SELECT {intent}")
        else:
            raise ValueError(f"Unsupported SSVEP intent: {intent!r}")
        return self.state

    def reset(self) -> SelectionState:
        self._index = 0
        self._selected = None
        self._history.clear()
        return self.state
