"""Deterministic scripted pick controller for Step 3; no learned policy is used."""

from __future__ import annotations

from simulation.environment import RobotSimulation


class ScriptedRobotController:
    """Move a visible gripper, attach the selected object, and lift it."""

    def __init__(self, simulation: RobotSimulation) -> None:
        self.simulation = simulation

    def _move_gripper(self, position: tuple[float, float, float], attached: str | None = None, steps: int = 40) -> None:
        start = self.simulation.gripper_position()
        for index in range(1, steps + 1):
            fraction = index / steps
            current = tuple(start[i] + fraction * (position[i] - start[i]) for i in range(3))
            self.simulation.set_gripper(current)
            if attached is not None:
                self.simulation.attach_to_gripper(attached)
            self.simulation.step()

    def execute(self, action: str, target: str) -> dict[str, object]:
        if action != "pick":
            raise ValueError(f"Unsupported scripted action: {action!r}")
        if target not in self.simulation.objects:
            raise ValueError(f"Unknown target object: {target!r}")
        initial_z = self.simulation.position(target)[2]
        object_position = self.simulation.position(target)
        above = (object_position[0], object_position[1], 0.78)
        grasp = (object_position[0], object_position[1], 0.65)
        self._move_gripper(above)
        self._move_gripper(grasp)
        grasp_success = sum((a - b) ** 2 for a, b in zip(self.simulation.gripper_position(), grasp)) ** 0.5 < 0.02
        if grasp_success:
            self.simulation.attach_to_gripper(target)
        self._move_gripper((object_position[0], object_position[1], 0.98), attached=target)
        final_z = self.simulation.position(target)[2]
        lifted = final_z > initial_z + 0.15
        return {
            "action": action,
            "target": target,
            "initial_z": initial_z,
            "final_z": final_z,
            "lift_delta": final_z - initial_z,
            "grasp_success": grasp_success,
            "lifted": lifted,
        }
