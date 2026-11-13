import numpy as np
import pytest

from experiments.ssvep.decoder import CCAConfig, FBCCAConfig, SSVEPDecoder, _reference_signals


def test_reference_shape_and_harmonics():
    reference = _reference_signals(10.0, 500, CCAConfig(harmonics=3))
    assert reference.shape == (500, 6)


def test_cca_decodes_synthetic_signal_for_unit_validation():
    config = CCAConfig(sampling_rate_hz=250, harmonics=2)
    reference = _reference_signals(10.0, 500, config)
    eeg = reference[:, :2].T + 0.01 * np.random.default_rng(4).normal(size=(2, 500))
    result = SSVEPDecoder([8.0, 10.0, 12.0], config).predict(eeg)
    assert result.frequency_hz == 10.0
    assert result.score_margin > 0


def test_fbcca_rejects_filter_at_nyquist():
    config = FBCCAConfig(bands_hz=((6.0, 125.0),))
    with pytest.raises(ValueError):
        SSVEPDecoder([10.0], config).score(np.ones((4, 500)))


def test_decoder_rejects_nonfinite_input():
    eeg = np.zeros((4, 500))
    eeg[0, 0] = np.nan
    with pytest.raises(ValueError):
        SSVEPDecoder([10.0]).predict(eeg)
