"""Deterministic semantic conversion for Step 4."""

from __future__ import annotations


class IntentAdapter:
    """Convert a selected object into the language instruction sent to GR00T."""

    def convert(self, event: str | dict[str, object]) -> str:
        if isinstance(event, str):
            target = event
        else:
            target = event.get("target") or event.get("selected_object")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("A selected target is required to create an instruction")
        target = target.strip().lower()
        if target not in {"apple", "cup", "phone"}:
            raise ValueError(f"Unsupported target object: {target!r}")
        return f"Pick up the {target}."

    def convert_libero(self, event: str | dict[str, object]) -> str:
        """Map the demo selection to an explicit official LIBERO task.

        LIBERO's official object suite does not contain a cup. The mapping is
        explicit so the Step 4 demo cannot silently pretend that its simulated
        object is the same as the UI's cup.
        """
        if isinstance(event, str):
            target = event
        else:
            target = event.get("target") or event.get("selected_object")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("A selected target is required for LIBERO")
        target = target.strip().lower()
        if target == "cup":
            return "Pick up the alphabet soup and place it in the basket."
        raise ValueError(f"No explicit LIBERO demo mapping exists for target {target!r}")
