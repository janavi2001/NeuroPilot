from pathlib import Path

import numpy as np
import pytest

from bci.ssvep_decoder import UNSURE, SSVEPDecoder
from interface.ssvep_selector import SSVEPSelector, load_ssvep_config


CONFIG = load_ssvep_config()
FS = 250.0
SAMPLES = int(CONFIG.window_seconds * FS)


def make_decoder(margin_threshold: float = 0.0) -> SSVEPDecoder:
    return SSVEPDecoder(CONFIG.targets, len(CONFIG.channels), FS, CONFIG.window_seconds, CONFIG.harmonics, margin_threshold)


def synthetic_ssvep(frequency: float, seed: int = 0, amplitude: float = 1.0) -> np.ndarray:
    """Unit-test signal only: a sinusoid buried in channel noise."""
    rng = np.random.default_rng(seed)
    time = np.arange(SAMPLES) / FS
    signal = amplitude * np.sin(2 * np.pi * frequency * time)
    gains = rng.uniform(0.5, 1.5, size=(len(CONFIG.channels), 1))
    return gains * signal + rng.normal(0, 1.0, size=(len(CONFIG.channels), SAMPLES))


def test_config_has_four_distinct_targets_and_cancel():
    assert list(CONFIG.targets) == ["APPLE", "CUP", "PHONE", "CANCEL"]
    assert len(set(CONFIG.targets.values())) == 4
    assert CONFIG.objects == ("APPLE", "CUP", "PHONE")
    assert CONFIG.margin_threshold >= 0


@pytest.mark.parametrize("name", ["APPLE", "CUP", "PHONE", "CANCEL"])
def test_decoder_identifies_each_target_frequency(name):
    prediction = make_decoder().predict(synthetic_ssvep(CONFIG.targets[name]))
    assert prediction["intent"] == name
    assert prediction["accepted"] is True
    assert prediction["margin"] > 0
    assert set(prediction["scores"]) == set(CONFIG.targets)


def test_decoder_is_deterministic():
    decoder = make_decoder()
    window = synthetic_ssvep(CONFIG.targets["CUP"], seed=3)
    assert decoder.predict(window) == decoder.predict(window.copy())


def test_low_margin_returns_unsure_without_dropping_best_guess():
    prediction = make_decoder(margin_threshold=1.0).predict(synthetic_ssvep(CONFIG.targets["CUP"]))
    assert prediction["intent"] == UNSURE
    assert prediction["accepted"] is False
    assert prediction["best_target"] == "CUP"


@pytest.mark.parametrize(
    "window, error",
    [
        (np.zeros((8, 10)), ValueError),
        (np.zeros((3, SAMPLES)), ValueError),
        (np.ones((8, SAMPLES), dtype=np.int64), TypeError),
        (np.full((8, SAMPLES), np.nan), ValueError),
        ([[0.0] * SAMPLES] * 8, TypeError),
    ],
)
def test_decoder_rejects_invalid_input(window, error):
    with pytest.raises(error):
        make_decoder().predict(window)


def test_decoder_rejects_flat_channel():
    window = synthetic_ssvep(CONFIG.targets["APPLE"])
    window[2] = 0.0
    with pytest.raises(ValueError, match="flat channel"):
        make_decoder().predict(window)


def test_decoder_rejects_bad_target_definitions():
    with pytest.raises(ValueError):
        SSVEPDecoder({"A": 10.0}, 8)
    with pytest.raises(ValueError):
        SSVEPDecoder({"A": 10.0, "B": 10.0}, 8)
    with pytest.raises(ValueError):
        SSVEPDecoder({"A": 10.0, UNSURE: 12.0}, 8)


def test_selector_select_cancel_and_unsure():
    selector = SSVEPSelector(CONFIG.objects, CONFIG.cancel_target)
    assert selector.apply({"intent": "PHONE"}).selected_object == "PHONE"
    assert selector.apply({"intent": UNSURE}).selected_object == "PHONE"
    assert selector.apply({"intent": "CANCEL"}).selected_object is None
    state = selector.apply({"intent": "CUP"})
    assert state.selected_object == "CUP"
    assert state.current_object == "CUP"
    assert state.history == (
        "LOOK PHONE -> SELECT PHONE",
        "UNSURE -> no change",
        "CANCEL -> selection cleared",
        "LOOK CUP -> SELECT CUP",
    )
    with pytest.raises(ValueError, match="Unsupported"):
        selector.apply({"intent": "LEFT"})
    assert selector.reset().selected_object is None


def _beta_cached() -> bool:
    try:
        import mne
        import moabb  # noqa: F401
    except ImportError:
        return False
    root = Path(mne.get_config("MNE_DATA") or Path.home() / "mne_data")
    return (root / "MNE-liu2020beta-data" / "S1.mat").is_file()


@pytest.mark.skipif(not _beta_cached(), reason="real BETA S1 data is not downloaded")
def test_real_beta_selection_drives_mujoco_pick():
    from bci.ssvep_data import load_beta_trials
    from robot.scripted_controller import ScriptedRobotController
    from simulation.environment import RobotSimulation

    eeg, labels, metadata = load_beta_trials(1, CONFIG.channels, CONFIG.window_seconds, list(CONFIG.targets.values()))
    assert eeg.shape == (16, 8, SAMPLES)
    assert metadata["sampling_rate_hz"] == FS
    names = {frequency: name for name, frequency in CONFIG.targets.items()}
    truths = [names[float(label)] for label in labels]
    assert sorted(set(truths)) == sorted(CONFIG.targets)

    decoder = make_decoder(CONFIG.margin_threshold)
    predictions = [decoder.predict(window) for window in eeg]
    accuracy = np.mean([p["best_target"] == t for p, t in zip(predictions, truths)])
    assert accuracy >= 0.75
    assert decoder.predict(eeg[0]) == predictions[0]

    selector = SSVEPSelector(CONFIG.objects, CONFIG.cancel_target)
    cup_trial = truths.index("CUP")
    state = selector.apply(predictions[cup_trial])
    assert state.selected_object == "CUP"

    simulation = RobotSimulation(gui=False)
    try:
        result = ScriptedRobotController(simulation).execute("pick", state.selected_object)
        assert result["lifted"] is True
    finally:
        simulation.close()
