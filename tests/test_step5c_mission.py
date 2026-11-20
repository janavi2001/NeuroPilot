import numpy as np
import pytest

from interface.visualization.look_track import LookTrack
from interface.visualization.mission_renderer import STAGES, MissionRenderer, MissionState

TARGETS = {"BOWL": 8.6, "PLATE": 9.8, "CHEESE": 11.0, "WINE": 12.2, "STOVE": 14.0, "CANCEL": 15.8}
CHANNELS = ["O1", "Oz", "O2", "PO3", "PO4", "PO7", "PO8", "POz"]


@pytest.fixture(scope="module")
def renderer():
    r = MissionRenderer(TARGETS, CHANNELS, 0.075)
    yield r
    r.close()


def _track():
    rng = np.random.default_rng(0)
    times = np.arange(1, 51) / 25
    return LookTrack("BOWL", 1, rng.standard_normal((8, 500)), rng.standard_normal((8, 500)), 250.0, TARGETS,
                     {"intent": "BOWL", "margin": 0.2}, times, rng.random((50, 6)), decision_seconds=1.1)


@pytest.mark.parametrize("scene", ["kitchen", "worlds"])
def test_every_scene_renders_a_full_hd_frame(renderer, scene):
    image = np.zeros((64, 64, 3), np.uint8)
    state = MissionState(scene=scene, command=1, intended="BOWL", look=_track(), look_elapsed=0.6, live=True,
                         stage=len(STAGES), robot_frame=image, sim_frame=image, instruction="put the bowl on the stove",
                         banner="decoded BOWL", history=["#1 saw BOWL → BOWL"], stats={"a": "1"},
                         world_frames={k: image for k in ("a", "b", "c", "d")}, card=("T", "S", ["line", "· note ·"]))
    frame = renderer.frame(state)
    assert frame.shape == (1080, 1920, 3) and frame.dtype == np.uint8
    assert frame.std() > 1  # something was drawn
