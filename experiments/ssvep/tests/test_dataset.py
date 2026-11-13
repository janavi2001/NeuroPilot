import numpy as np
import pytest

from experiments.ssvep.dataset import SSVEPDataset, crop_window, select_channels


def sample_dataset():
    return SSVEPDataset(
        data=np.zeros((2, 4, 750)),
        frequencies_hz=np.array([8.0, 10.0]),
        phases_rad=np.zeros(2),
        channel_names=("O1", "Oz", "O2", "POz"),
        sampling_rate_hz=250.0,
        subject=1,
        stimulus_onset_seconds=0.5,
        stimulus_duration_seconds=2.0,
        metadata={},
    )


def test_select_channels_and_crop_window():
    dataset = sample_dataset()
    assert select_channels(dataset, dataset.channel_names) == (0, 1, 2, 3)
    assert crop_window(dataset, 1.0).shape == (2, 4, 250)


def test_crop_window_rejects_subject_duration_overrun():
    with pytest.raises(ValueError):
        crop_window(sample_dataset(), 3.0)
