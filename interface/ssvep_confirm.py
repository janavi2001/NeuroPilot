"""Two-look confirmation: an object is acted on only after two agreeing confident looks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bci.ssvep_decoder import UNSURE


@dataclass(frozen=True)
class ConfirmState:
    candidate: str | None  # previewed, waiting for a confirming look
    confirmed: str | None  # approved for the robot
    event: str  # what the last look did


class ConfirmingSelector:
    """First confident look previews an object; a second look at the same object confirms it.

    A look at a different object replaces the candidate, CANCEL clears everything,
    and UNSURE leaves the state unchanged. On held-out BETA subjects this cut wrong
    robot actions from 7.5% to 1.2% (scripts/step4c_confirm_eval.py).
    """

    def __init__(self, objects: tuple[str, ...], cancel_target: str = "CANCEL") -> None:
        if not objects or cancel_target in objects:
            raise ValueError("objects must be non-empty and must not include the cancel target")
        self.objects = tuple(name.upper() for name in objects)
        self.cancel_target = cancel_target.upper()
        self.state = ConfirmState(None, None, "start")

    def apply(self, prediction: dict[str, Any]) -> ConfirmState:
        intent = str(prediction["intent"]).upper()
        candidate = self.state.candidate
        if intent == UNSURE:
            self.state = ConfirmState(candidate, None, "unsure")
        elif intent == self.cancel_target:
            self.state = ConfirmState(None, None, "cancel")
        elif intent not in self.objects:
            raise ValueError(f"Unsupported SSVEP intent: {intent!r}")
        elif intent == candidate:
            self.state = ConfirmState(None, intent, "confirm")
        else:
            self.state = ConfirmState(intent, None, "candidate" if candidate is None else "switch")
        return self.state

    def reset(self) -> ConfirmState:
        self.state = ConfirmState(None, None, "start")
        return self.state
