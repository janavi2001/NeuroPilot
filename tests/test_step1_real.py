from pathlib import Path

import numpy as np
import pytest

from bci.decoder import BCIDecoder
from bci.preprocessing import load_real_trials


ROOT = Path(__file__).resolve().parents[1]
EEG = ROOT / "data/raw/MI_BCIC_IV2a_mat/A01T.mat"
CHECKPOINT = ROOT / "artifacts/checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt"


@pytest.mark.skipif(not EEG.is_file() or not CHECKPOINT.is_file(), reason="real Step 1 artifacts are not downloaded")
def test_real_trials_have_expected_shape_and_repeatable_prediction():
    eeg, labels, metadata = load_real_trials(EEG, limit=2)
    assert eeg.shape == (2, 65, 800)
    assert metadata["channels"] == 22
    assert set(labels) <= {"LEFT", "RIGHT"}

    decoder = BCIDecoder(CHECKPOINT)
    first = decoder.predict(eeg[0])
    second = decoder.predict(eeg[0])
    assert first["intent"] in {"LEFT", "RIGHT"}
    assert first == second
    assert np.isfinite(list(first["scores"].values())).all()
