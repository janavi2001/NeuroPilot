import numpy as np
import pytest

from interface.replay import RecordedEEGReplay
from interface.selector import ObjectSelector


def test_selector_right_right_left_selects_phone():
    selector = ObjectSelector()
    selector.apply("RIGHT")
    selector.apply("RIGHT")
    state = selector.apply("LEFT")
    assert state.current_object == "PHONE"
    assert state.selected_object == "PHONE"
    assert state.history == ("RIGHT -> CUP", "RIGHT -> PHONE", "LEFT -> SELECT PHONE")


def test_selector_wraps_and_rejects_unknown_intent():
    selector = ObjectSelector()
    selector.apply("RIGHT")
    selector.apply("RIGHT")
    assert selector.apply("RIGHT").current_object == "APPLE"
    with pytest.raises(ValueError, match="Unsupported"):
        selector.apply("UP")


def test_selector_reset_returns_to_apple():
    selector = ObjectSelector()
    selector.apply("RIGHT")
    selector.apply("LEFT")
    state = selector.reset()
    assert state.current_object == "APPLE"
    assert state.selected_object is None
    assert state.history == ()


def test_recorded_replay_calls_decoder_in_order():
    class FakeDecoder:
        def __init__(self):
            self.calls = []

        def predict(self, eeg_window):
            self.calls.append(float(eeg_window[0, 0]))
            return {"intent": "RIGHT", "score": 0.5, "scores": {"LEFT": 0.1, "RIGHT": 0.5}}

    eeg = np.zeros((2, 65, 800), dtype=np.float32)
    eeg[0, 0, 0] = 1
    eeg[1, 0, 0] = 2
    decoder = FakeDecoder()
    events = list(RecordedEEGReplay(eeg, np.array(["LEFT", "RIGHT"]), decoder))
    assert [event.trial_number for event in events] == [1, 2]
    assert [event.ground_truth for event in events] == ["LEFT", "RIGHT"]
    assert decoder.calls == [1.0, 2.0]
