import mujoco
import numpy as np
import pytest

from bci.ssvep_decoder import UNSURE
from interface.ssvep_replay import intended_final, replay_sequence
from interface.ssvep_selector import SSVEPSelector
from simulation.franka_environment import GRIPPER_OPEN, PANDA_XML


requires_franka = pytest.mark.skipif(not PANDA_XML.is_file(), reason="Franka model not fetched; run scripts/fetch_franka.sh")


@requires_franka
@pytest.mark.parametrize("target", ["APPLE", "CUP", "PHONE"])
def test_franka_physically_grasps_and_lifts_each_object(target):
    from robot.franka_controller import FrankaPickController
    from simulation.franka_environment import FrankaSimulation

    simulation = FrankaSimulation()
    result = FrankaPickController(simulation).execute("pick", target)
    assert result["grasp_success"] is True
    assert result["fingers_in_contact_after_lift"] is True
    assert result["lift_delta"] > 0.15
    assert result["object_tcp_distance"] < 0.04
    assert result["max_other_object_shift"] < 0.01
    assert result["lifted"] is True


@requires_franka
def test_lift_requires_closing_the_gripper():
    """Negative control: identical motion with the gripper held open must not lift the cup."""
    from robot.franka_controller import FrankaPickController
    from simulation.franka_environment import FrankaSimulation

    class OpenGripperController(FrankaPickController):
        def _hold(self, phase, seconds, gripper):
            super()._hold(phase, seconds, GRIPPER_OPEN)

        def _move(self, phase, position, rotation, seconds, gripper):
            super()._move(phase, position, rotation, seconds, GRIPPER_OPEN)

    result = OpenGripperController(FrankaSimulation()).execute("pick", "CUP")
    assert result["lift_delta"] < 0.02
    assert result["lifted"] is False


@requires_franka
def test_objects_rest_without_robot_contact():
    from simulation.franka_environment import FrankaSimulation

    simulation = FrankaSimulation()
    before = {name: simulation.position(name) for name in simulation.OBJECTS}
    simulation.step(1000)
    for name, position in before.items():
        assert np.linalg.norm(simulation.position(name) - position) < 1e-3


@requires_franka
def test_ik_reaches_target_pose():
    from robot.franka_controller import FrankaPickController
    from simulation.franka_environment import FrankaSimulation

    simulation = FrankaSimulation()
    controller = FrankaPickController(simulation)
    rotation = controller._grasp_rotation(np.array([0.0, 1.0]))
    target = np.array([0.5, 0.1, 0.2])
    q = controller._solve_ik(target, rotation, simulation.data.qpos[:7].copy(), iterations=200)
    data = mujoco.MjData(simulation.model)
    data.qpos[:] = simulation.data.qpos
    data.qpos[:7] = q
    mujoco.mj_kinematics(simulation.model, data)
    assert np.linalg.norm(data.site_xpos[simulation.tcp_site] - target) < 1e-3
    assert np.allclose(data.site_xmat[simulation.tcp_site].reshape(3, 3), rotation, atol=1e-2)


@requires_franka
def test_franka_controller_rejects_invalid_requests():
    from robot.franka_controller import FrankaPickController
    from simulation.franka_environment import FrankaSimulation

    controller = FrankaPickController(FrankaSimulation())
    with pytest.raises(ValueError, match="Unsupported action"):
        controller.execute("throw", "CUP")
    with pytest.raises(ValueError, match="Unknown target"):
        controller.execute("pick", "CANCEL")


def test_replay_chooses_trials_by_ground_truth_and_retries_unsure():
    class ScriptedDecoder:
        """Returns a fixed answer per trial index so the replay logic can be checked."""
        answers = {0: "APPLE", 1: UNSURE, 2: "CUP", 3: "CANCEL"}

        def predict(self, window):
            intent = self.answers[int(window[0, 0])]
            return {"intent": intent, "margin": 0.1, "score": 0.5}

    eeg = np.arange(4, dtype=float).reshape(4, 1, 1) * np.ones((4, 1, 1))
    truths = ["APPLE", "CUP", "CUP", "CANCEL"]
    selector = SSVEPSelector(("APPLE", "CUP", "PHONE"))
    events = replay_sequence(ScriptedDecoder(), selector, eeg, truths, ["CUP", "CANCEL", "APPLE"])
    assert [(e.intended, e.trial_number, e.prediction["intent"]) for e in events] == [
        ("CUP", 2, UNSURE), ("CUP", 3, "CUP"), ("CANCEL", 4, "CANCEL"), ("APPLE", 1, "APPLE"),
    ]
    assert selector.state.selected_object == "APPLE"
    assert intended_final(["CUP", "CANCEL"], "CANCEL") is None
    assert intended_final(["PHONE", "CANCEL", "CUP"], "CANCEL") == "CUP"


@requires_franka
def test_real_beta_sequence_drives_franka_pick():
    from tests.test_step4b_ssvep import CONFIG, _beta_cached, make_decoder

    if not _beta_cached():
        pytest.skip("real BETA S1 data is not downloaded")
    from bci.ssvep_data import load_beta_trials
    from robot.franka_controller import FrankaPickController
    from simulation.franka_environment import FrankaSimulation

    eeg, labels, _ = load_beta_trials(1, CONFIG.channels, CONFIG.window_seconds, list(CONFIG.targets.values()))
    names = {frequency: name for name, frequency in CONFIG.targets.items()}
    truths = [names[float(label)] for label in labels]
    selector = SSVEPSelector(CONFIG.objects, CONFIG.cancel_target)
    replay_sequence(make_decoder(CONFIG.margin_threshold), selector, eeg, truths, ["PHONE", "CANCEL", "CUP"])
    assert selector.state.selected_object == "CUP"
    result = FrankaPickController(FrankaSimulation()).execute("pick", selector.state.selected_object)
    assert result["lifted"] is True
