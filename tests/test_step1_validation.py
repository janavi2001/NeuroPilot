import numpy as np
import pytest

from bci.decoder import BCIDecoder


def test_decoder_requires_leaf_shape(tmp_path):
    checkpoint = tmp_path / "missing.ckpt"
    with pytest.raises(FileNotFoundError):
        BCIDecoder(checkpoint)


def test_prediction_input_validation_without_loading_model():
    decoder = object.__new__(BCIDecoder)
    with pytest.raises(ValueError, match="Expected LEAF input shape"):
        decoder.predict(np.zeros((22, 800), dtype=np.float32))

    with pytest.raises(TypeError, match="floating-point"):
        decoder.predict(np.zeros((65, 800), dtype=np.int16))

    with pytest.raises(ValueError, match="non-finite"):
        values = np.zeros((65, 800), dtype=np.float32)
        values[0, 0] = np.nan
        decoder.predict(values)
