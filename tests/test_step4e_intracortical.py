from dataclasses import dataclass

import numpy as np
import pytest

from bci.intracortical_decoder import DecoderWindow, ReachIntentDecoder
from bci.nlb_data import DEFAULT_NWB, DIRECTION_TARGETS, direction_class

requires_nlb = pytest.mark.skipif(not DEFAULT_NWB.is_file(), reason="NLB MC_Maze_Large not downloaded; run scripts/fetch_nlb_mc_maze.sh")


@pytest.mark.parametrize("xy, expected", [
    ((-130, -13), "APPLE"), ((-126, 15), "APPLE"), ((-105, 76), "APPLE"), ((12, 84), "CUP"),
    ((103, 83), "PHONE"), ((123, 69), "PHONE"), ((123, 71), "PHONE"), ((123, -81), "CANCEL"), ((124, -79), "CANCEL"),
])
def test_all_nine_maze_targets_map_to_a_direction(xy, expected):
    assert direction_class(np.array(xy, dtype=float)) == expected


def test_direction_below_the_screen_is_rejected():
    with pytest.raises(ValueError, match="no direction class"):
        direction_class(np.array([0.0, -100.0]))


@dataclass
class FakeTrials:
    """Unit-test stand-in: each class drives a different block of units."""
    labels: np.ndarray
    seed: int = 0

    def spike_counts(self, trial, start, stop, bin_seconds):
        rng = np.random.default_rng(self.seed + trial)
        bins = int(round((stop - start) / bin_seconds))
        rates = np.full((20, bins), 2.0)
        block = DIRECTION_TARGETS.index(self.labels[trial])
        rates[block * 5:(block + 1) * 5] = 8.0
        return rng.poisson(rates).astype(float)


def test_decoder_learns_separable_classes():
    labels = np.array([DIRECTION_TARGETS[i % 4] for i in range(80)])
    trials = FakeTrials(labels)
    decoder = ReachIntentDecoder(DIRECTION_TARGETS, DecoderWindow(-0.3, 0.0, 0.1)).fit(trials, range(60))
    predictions = [decoder.predict_trial(trials, i)["intent"] for i in range(60, 80)]
    assert np.mean(np.array(predictions) == labels[60:]) >= 0.95
    result = decoder.predict_trial(trials, 60)
    assert set(result["scores"]) == set(DIRECTION_TARGETS)
    assert result["score"] == max(result["scores"].values())
    assert abs(sum(result["scores"].values()) - 1) < 1e-6


def test_decoder_rejects_invalid_use():
    labels = np.array([DIRECTION_TARGETS[i % 4] for i in range(40)])
    decoder = ReachIntentDecoder(DIRECTION_TARGETS, DecoderWindow(-0.3, 0.0, 0.1))
    with pytest.raises(RuntimeError, match="not fitted"):
        decoder.predict_features(np.zeros(60))
    with pytest.raises(ValueError, match="no trials"):
        ReachIntentDecoder(DIRECTION_TARGETS, DecoderWindow(-0.3, 0.0, 0.1)).fit(FakeTrials(labels), [0, 1, 2])
    decoder.fit(FakeTrials(labels), range(40))
    with pytest.raises(ValueError, match="features"):
        decoder.predict_features(np.zeros(7))
    with pytest.raises(ValueError, match="non-finite"):
        decoder.predict_features(np.full(60, np.nan))
    with pytest.raises(ValueError, match="Invalid decoder window"):
        ReachIntentDecoder(DIRECTION_TARGETS, DecoderWindow(0.0, -0.3, 0.1))


@requires_nlb
def test_real_data_loads_with_official_split():
    from bci.nlb_data import load_mc_maze

    trials = load_mc_maze()
    assert trials.n_units == 162 and len(trials.labels) == 500
    assert (trials.split == "train").sum() == 375 and (trials.split == "val").sum() == 125
    assert set(trials.labels) == set(DIRECTION_TARGETS)
    assert trials.spike_counts(0, -0.4, 0.0, 0.1).shape == (162, 4)
    assert np.linalg.norm(trials.cursor_at(0, -0.3)) < 20  # cursor still at centre before movement


@requires_nlb
def test_real_decoder_generalises_to_held_out_trials_and_shuffle_control_fails():
    from bci.nlb_data import load_mc_maze

    trials = load_mc_maze()
    train, val = np.flatnonzero(trials.split == "train"), np.flatnonzero(trials.split == "val")
    window = DecoderWindow(-0.4, 0.0, 0.1)
    decoder = ReachIntentDecoder(DIRECTION_TARGETS, window).fit(trials, train)
    accuracy = np.mean([decoder.predict_trial(trials, i)["intent"] == trials.labels[i] for i in val])
    assert accuracy >= 0.9
    shuffled = trials.__class__(**{**trials.__dict__, "labels": trials.labels.copy()})
    shuffled.labels[train] = np.random.default_rng(1).permutation(trials.labels[train])
    control = ReachIntentDecoder(DIRECTION_TARGETS, window).fit(shuffled, train)
    assert np.mean([control.predict_trial(trials, i)["intent"] == trials.labels[i] for i in val]) < 0.5
