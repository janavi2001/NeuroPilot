"""Safety decisions for rented Vast.ai GPUs.

Kept separate from the CLI calls so every decision can be unit-tested. The
rule that failed on 2026-11-14 (a text grep that mistook "could not read the
instance list" for "instance is gone") is replaced by strict JSON parsing:
anything unreadable is UNKNOWN, and UNKNOWN never counts as gone.
"""

from __future__ import annotations

import json

UNKNOWN = None

WATCH = "watch"
GONE = "gone"
DESTROY_DEADLINE = "destroy:deadline"
DESTROY_LOW_CREDIT = "destroy:low-credit"

DONE, RUNNING, FAILED, UNREACHABLE = "DONE", "RUNNING", "FAILED", "UNREACHABLE"
SSH_CONNECTION_ERROR = 255


class VastOutputError(ValueError):
    pass


def parse_instance_ids(text: str) -> set[int]:
    """IDs in ``vastai show instances --raw`` output; raises if it is not that output."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as error:
        raise VastOutputError(f"not JSON: {str(text)[:120]!r}") from error
    if not isinstance(data, list):
        raise VastOutputError(f"expected a JSON list of instances, got {type(data).__name__}")
    try:
        return {int(item["id"]) for item in data}
    except (KeyError, TypeError, ValueError) as error:
        raise VastOutputError("instance entry without an integer id") from error


def instance_present(text: str | None, instance_id: int) -> bool | None:
    """True/False when the listing is readable, UNKNOWN otherwise."""
    if text is None:
        return UNKNOWN
    try:
        return instance_id in parse_instance_ids(text)
    except VastOutputError:
        return UNKNOWN


def parse_credit(text: str | None) -> float | None:
    """Account credit from ``vastai show user --raw``, UNKNOWN if unreadable."""
    try:
        return float(json.loads(text)["credit"])
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        return UNKNOWN


def guard_action(now: float, deadline: float, present: bool | None, credit: float | None,
                 min_credit: float) -> str:
    """What the watchdog should do this tick. Only a readable listing without the instance is GONE."""
    if present is False:
        return GONE
    if now >= deadline:
        return DESTROY_DEADLINE
    if credit is not None and credit < min_credit:
        return DESTROY_LOW_CREDIT
    return WATCH


def classify_remote(returncode: int, stdout: str) -> str:
    """Map one ssh status probe to DONE / RUNNING / FAILED / UNREACHABLE.

    The remote probe prints exactly DONE, RUNNING or STOPPED. STOPPED (the job
    exited without writing its done marker) is a failure. A dropped connection
    or any other output is UNREACHABLE, which is never treated as progress.
    """
    lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
    last = lines[-1] if lines else ""
    if returncode == SSH_CONNECTION_ERROR or last not in {"DONE", "RUNNING", "STOPPED"}:
        return UNREACHABLE
    return {"DONE": DONE, "RUNNING": RUNNING, "STOPPED": FAILED}[last]


class RemoteWait:
    """Counts consecutive UNREACHABLE probes; gives up after ``max_unreachable``."""

    def __init__(self, max_unreachable: int) -> None:
        if max_unreachable < 1:
            raise ValueError("max_unreachable must be at least 1")
        self.max_unreachable = max_unreachable
        self.unreachable = 0

    def update(self, state: str) -> str | None:
        """Return a final state (DONE/FAILED/UNREACHABLE) or None to keep waiting."""
        if state == UNREACHABLE:
            self.unreachable += 1
            return UNREACHABLE if self.unreachable >= self.max_unreachable else None
        self.unreachable = 0
        return None if state == RUNNING else state
