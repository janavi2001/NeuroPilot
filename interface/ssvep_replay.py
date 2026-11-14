"""Replay recorded SSVEP trials for an intended look sequence."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from bci.ssvep_decoder import UNSURE, SSVEPDecoder
from .ssvep_selector import SSVEPSelector


@dataclass(frozen=True)
class LookEvent:
    intended: str
    trial_number: int  # 1-based index into the subject's recorded trials
    prediction: dict[str, Any]
    selected_object: str | None
    latency_ms: float


def replay_sequence(
    decoder: SSVEPDecoder,
    selector: SSVEPSelector,
    eeg: np.ndarray,
    truths: Sequence[str],
    sequence: Sequence[str],
) -> list[LookEvent]:
    """For each intended target, decode that target's next unused recorded trial.

    Trials are chosen by ground truth (what the subject was looking at), never by
    whether they decode correctly. An ``UNSURE`` result moves on to the next
    recorded trial of the same target, like a user looking again.
    """
    used: set[int] = set()
    events = []
    for intended in sequence:
        for index, truth in enumerate(truths):
            if truth != intended or index in used:
                continue
            used.add(index)
            start = time.perf_counter()
            prediction = decoder.predict(eeg[index])
            latency_ms = 1000 * (time.perf_counter() - start)
            state = selector.apply(prediction)
            events.append(LookEvent(intended, index + 1, prediction, state.selected_object, latency_ms))
            if prediction["intent"] != UNSURE:
                break
    return events


def intended_final(sequence: Sequence[str], cancel_target: str) -> str | None:
    """The object the subject meant to end with: the last look, unless it was CANCEL."""
    return None if not sequence or sequence[-1] == cancel_target else sequence[-1]
