import numpy as np

from simulation.environment import RobotSimulation


def test_gr00t_observation_contract_shapes():
    simulation = RobotSimulation(gui=False)
    try:
        observation = simulation.gr00t_observation("Pick up the cup.")
        assert observation["video"]["exterior_image_1_left"].shape == (1, 2, 224, 224, 3)
        assert observation["video"]["exterior_image_1_left"].dtype == np.uint8
        assert observation["state"]["eef_9d"].shape == (1, 1, 9)
        assert observation["state"]["joint_position"].shape == (1, 1, 7)
    finally:
        simulation.close()
