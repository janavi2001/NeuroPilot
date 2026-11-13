from simulation.environment import RobotSimulation
from robot.scripted_controller import ScriptedRobotController


def test_scripted_controller_programmatically_lifts_cup():
    simulation = RobotSimulation(gui=False)
    try:
        result = ScriptedRobotController(simulation).execute("pick", "CUP")
        assert result["grasp_success"] is True
        assert result["lifted"] is True
        assert result["final_z"] > result["initial_z"] + 0.15
    finally:
        simulation.close()
