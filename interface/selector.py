"""Deterministic object selection driven by LEFT/RIGHT BCI events."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionState:
    objects: tuple[str, ...]
    current_index: int
    selected_object: str | None
    history: tuple[str, ...]

    @property
    def current_object(self) -> str:
        return self.objects[self.current_index]


class ObjectSelector:
    """Map RIGHT to next-object and LEFT to select-current-object."""

    def __init__(self, objects: tuple[str, ...] = ("APPLE", "CUP", "PHONE")) -> None:
        if not objects or any(not object_name for object_name in objects):
            raise ValueError("objects must be a non-empty tuple of names")
        self._objects = tuple(name.upper() for name in objects)
        self._index = 0
        self._selected: str | None = None
        self._history: list[str] = []

    @property
    def state(self) -> SelectionState:
        return SelectionState(self._objects, self._index, self._selected, tuple(self._history))

    def apply(self, intent: str) -> SelectionState:
        intent = intent.upper()
        if intent == "RIGHT":
            self._index = (self._index + 1) % len(self._objects)
            self._history.append(f"RIGHT -> {self.current_object}")
        elif intent == "LEFT":
            self._selected = self.current_object
            self._history.append(f"LEFT -> SELECT {self._selected}")
        else:
            raise ValueError(f"Unsupported selector intent: {intent!r}")
        return self.state

    @property
    def current_object(self) -> str:
        return self._objects[self._index]

    def reset(self) -> SelectionState:
        """Return to the initial object and clear the selection history."""

        self._index = 0
        self._selected = None
        self._history.clear()
        return self.state
