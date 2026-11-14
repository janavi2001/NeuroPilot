import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bci.metrics import wolpaw_bits
from bci.ssvep_stopping import EarlyStoppingDecoder, StoppingRule, margins_over_time
from interface.ssvep_selector import DEFAULT_CONFIG, load_ssvep_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_ssvep_config()
FS = 250.0


def _synthetic(frequency, amplitude, seed=0):
    """Unit-test signal only."""
    rng = np.random.default_rng(seed)
    time = np.arange(500) / FS
    return amplitude * np.sin(2 * np.pi * frequency * time) + rng.normal(0, 1.0, size=(len(CONFIG.channels), 500))


def _decoder(rule):
    return EarlyStoppingDecoder(CONFIG.targets, len(CONFIG.channels), rule, FS, CONFIG.window_seconds,
                                CONFIG.harmonics, CONFIG.margin_threshold)


def test_rule_loaded_from_config_and_relaxes_over_time():
    rule = StoppingRule.from_yaml(DEFAULT_CONFIG)
    assert (rule.min_seconds, rule.base_margin, rule.exponent) == (0.3, 0.10, 1.5)
    assert rule.required_margin(2.0, 2.0) == pytest.approx(0.10)
    assert rule.required_margin(0.5, 2.0) == pytest.approx(0.8)
    assert rule.required_margin(1.0, 2.0) > rule.required_margin(1.5, 2.0)


def test_strong_signal_stops_early_with_correct_target():
    result = _decoder(StoppingRule(0.3, 0.10, 1.5)).predict(_synthetic(CONFIG.targets["PHONE"], 1.5))
    assert result["early"] is True and result["decision_seconds"] < 2.0
    assert result["intent"] == "PHONE"


def test_impossible_margin_falls_back_to_full_window():
    window = _synthetic(CONFIG.targets["CUP"], 1.0)
    result = _decoder(StoppingRule(0.3, 10.0, 0.0)).predict(window)
    assert result["early"] is False and result["decision_seconds"] == 2.0
    full = _decoder(StoppingRule(0.3, 10.0, 0.0)).full.predict(window)
    assert {k: result[k] for k in full} == full


def test_invalid_rule_and_input_are_rejected():
    with pytest.raises(ValueError):
        _decoder(StoppingRule(0.0, 0.1))
    with pytest.raises(ValueError):
        _decoder(StoppingRule(0.3, -0.1))
    with pytest.raises(ValueError):
        _decoder(StoppingRule(0.3, 0.1)).predict(np.zeros((8, 100)))


def test_metrics_wolpaw():
    assert wolpaw_bits(1.0, 4) == pytest.approx(2.0)
    assert wolpaw_bits(0.9, 4) == pytest.approx(1.3725, abs=1e-3)


def _beta_cached():
    try:
        import mne
        import moabb  # noqa: F401
    except ImportError:
        return False
    root = Path(mne.get_config("MNE_DATA") or Path.home() / "mne_data")
    return (root / "MNE-liu2020beta-data" / "S1.mat").is_file()


@pytest.mark.skipif(not _beta_cached(), reason="real BETA S1 data is not downloaded")
def test_production_decoder_matches_evaluation_simulation_on_real_trials():
    from bci.ssvep_data import load_beta_trials

    spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts" / "step4d_early_stop_eval.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    rule = StoppingRule.from_yaml(DEFAULT_CONFIG)
    decoder = _decoder(rule)
    names = list(CONFIG.targets)
    times = [round(t, 1) for t in np.arange(0.3, 2.0 - 1e-9, 0.1)] + [2.0]
    eeg, _, _ = load_beta_trials(1, CONFIG.channels, CONFIG.window_seconds, list(CONFIG.targets.values()))
    for window in eeg:
        curve, _ = margins_over_time(window, CONFIG.targets, FS, times, CONFIG.harmonics)
        index = ev.decide({"best": curve[0].astype(int), "margin": curve[1]}, times, rule)
        result = decoder.predict(window)
        assert result["decision_seconds"] == pytest.approx(times[index])
        assert result["best_target"] == names[int(curve[0][index])]
