import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bci.ssvep_decoder import UNSURE, SSVEPDecoder
from interface.ssvep_confirm import ConfirmingSelector
from interface.visualization.look_track import LookTrack, ssvep_amplitude, tracks_for_sequence
from simulation.franka_environment import PANDA_XML

ROOT = Path(__file__).resolve().parents[1]
requires_franka = pytest.mark.skipif(not PANDA_XML.is_file(), reason="Franka model not fetched; run scripts/fetch_franka.sh")
TARGETS = {"APPLE": 9.8, "CUP": 11.8, "PHONE": 13.8, "CANCEL": 15.8}
CHANNELS = ["O1", "Oz", "O2", "PO3", "PO4", "PO7", "PO8", "POz"]
FS = 250.0


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic(frequency, seed=0):
    """Unit-test signal only: a sinusoid in noise on every channel."""
    rng = np.random.default_rng(seed)
    time = np.arange(500) / FS
    return np.sin(2 * np.pi * frequency * time) + rng.normal(0, 1.0, size=(len(CHANNELS), 500))


def test_confirming_selector_needs_two_agreeing_looks():
    selector = ConfirmingSelector(("APPLE", "CUP", "PHONE"))
    assert selector.apply({"intent": "CUP"}).candidate == "CUP"
    assert selector.apply({"intent": UNSURE}).candidate == "CUP"
    state = selector.apply({"intent": "CUP"})
    assert (state.candidate, state.confirmed, state.event) == (None, "CUP", "confirm")


def test_confirming_selector_switch_and_cancel():
    selector = ConfirmingSelector(("APPLE", "CUP", "PHONE"))
    selector.apply({"intent": "APPLE"})
    state = selector.apply({"intent": "PHONE"})
    assert (state.candidate, state.confirmed, state.event) == ("PHONE", None, "switch")
    state = selector.apply({"intent": "CANCEL"})
    assert (state.candidate, state.confirmed, state.event) == (None, None, "cancel")
    with pytest.raises(ValueError, match="Unsupported"):
        selector.apply({"intent": "LEFT"})


def test_look_track_grows_window_and_matches_full_decoder():
    decoder = SSVEPDecoder(TARGETS, len(CHANNELS), FS, 2.0, 3, 0.025)
    eeg = _synthetic(TARGETS["CUP"])
    track = LookTrack.build("CUP", 1, eeg, eeg, decoder, fps=25)
    assert track.frame_scores.shape == (50, 4)
    assert np.isnan(track.frame_scores[:5]).all()  # < 0.25 s
    assert np.isfinite(track.frame_scores[6:]).all()
    assert track.final == decoder.predict(eeg)
    assert list(TARGETS)[int(np.argmax(track.scores_at(2.0)))] == "CUP"
    assert track.topomap_at(0.3) is None
    checkpoint, leader, values = track.topomap_at(2.0)
    assert checkpoint == 2.0 and leader == "CUP" and values.shape == (len(CHANNELS),)


def test_ssvep_amplitude_is_larger_on_channels_carrying_the_frequency():
    rng = np.random.default_rng(1)
    time = np.arange(500) / FS
    eeg = rng.normal(0, 0.2, size=(3, 500))
    eeg[0] += np.sin(2 * np.pi * 11.8 * time)
    amplitude = ssvep_amplitude(eeg, 11.8, FS)
    assert amplitude[0] > 3 * amplitude[1:].max()


def test_tracks_grouped_in_recording_order():
    assert tracks_for_sequence(["CUP", "APPLE", "CUP"]) == {"CUP": [0, 2], "APPLE": [1]}


def test_wolpaw_bits():
    module = _load_script("step4c_confirm_eval")
    assert module.wolpaw_bits(1.0, 4) == pytest.approx(2.0)
    assert module.wolpaw_bits(0.25, 4) == 0.0
    assert module.wolpaw_bits(0.9, 4) == pytest.approx(1.3725, abs=1e-3)


@requires_franka
def test_pick_aborts_before_grasp_and_returns_home():
    from robot.franka_controller import FrankaPickController
    from simulation.franka_environment import FrankaSimulation

    simulation = FrankaSimulation()
    controller = FrankaPickController(simulation)
    cup_before = simulation.position("CUP")
    calls = {"n": 0}

    def abort_after_one_second():
        calls["n"] += 1
        return calls["n"] > 500

    result = controller.execute("pick", "CUP", should_abort=abort_after_one_second)
    assert result["aborted"] is True and result["aborted_during"] == "approach"
    assert result["lifted"] is False
    assert np.linalg.norm(simulation.position("CUP") - cup_before) < 1e-3
    controller.go_home(2.0)
    simulation.step(250)
    assert np.abs(simulation.data.qpos[:7] - simulation.home_qpos).max() < 0.02


@requires_franka
def test_abort_is_ignored_once_the_gripper_closes():
    from robot.franka_controller import FrankaPickController
    from simulation.franka_environment import FrankaSimulation

    simulation = FrankaSimulation()
    controller = FrankaPickController(simulation)
    abort = {"armed": False}
    controller._on_step = lambda phase: abort.update(armed=abort["armed"] or phase == "grasp")
    result = controller.execute("pick", "CUP", should_abort=lambda: abort["armed"])
    assert result["aborted"] is False
    assert result["lifted"] is True


@requires_franka
def test_plan_grasp_places_tcp_at_grasp_point():
    import mujoco

    from robot.franka_controller import FrankaPickController
    from simulation.franka_environment import FrankaSimulation

    simulation = FrankaSimulation()
    q = FrankaPickController(simulation).plan_grasp("PHONE")
    data = mujoco.MjData(simulation.model)
    data.qpos[:] = simulation.data.qpos
    data.qpos[:7] = q
    mujoco.mj_kinematics(simulation.model, data)
    phone = simulation.position("PHONE")
    expected = np.array([phone[0], phone[1], simulation.object_specs["PHONE"].grasp_height])
    assert np.linalg.norm(data.site_xpos[simulation.tcp_site] - expected) < 2e-3
    assert np.abs(simulation.data.qpos[:7] - simulation.home_qpos).max() < 0.05  # planning did not move the robot


@requires_franka
def test_hero_frame_renders_all_panels():
    import mne

    from interface.visualization.hero_renderer import DemoState, HeroRenderer
    from simulation.franka_environment import FrankaSimulation

    info = mne.create_info(CHANNELS, FS, "eeg")
    info.set_montage("standard_1005")
    simulation = FrankaSimulation()
    decoder = SSVEPDecoder(TARGETS, len(CHANNELS), FS, 2.0, 3, 0.025)
    eeg = _synthetic(TARGETS["PHONE"])
    track = LookTrack.build("PHONE", 1, eeg, eeg, decoder, fps=25)
    renderer = HeroRenderer(simulation, TARGETS, CHANNELS, info, 0.025)
    try:
        state = DemoState(look=track, look_elapsed=1.2, candidate="PHONE",
                          ghost_qpos=simulation.home_qpos.copy(), banner="test", history=["#1 test"])
        frame = renderer.frame(state)
        assert frame.shape == (1080, 1920, 3) and frame.dtype == np.uint8
        state.card = ("TITLE", "subtitle", "footnote")
        assert renderer.frame(state).std() > 0
    finally:
        renderer.close()
