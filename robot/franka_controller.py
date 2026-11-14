"""Scripted pick for the Franka Panda: IK waypoints, joint position control, friction grasp.

The controller only writes actuator commands. The object is never attached or
teleported; whether it is lifted is decided by the physics and then measured.
"""

from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np

from simulation.franka_environment import ARM_JOINTS, GRIPPER_CLOSED, GRIPPER_OPEN, FrankaSimulation


class PickAborted(Exception):
    """Raised inside a pick when the abort callback fires before the gripper closes."""


ABORTABLE_PHASES = ("approach", "descend")


class FrankaPickController:
    """Approach from above, descend, close the gripper, and lift the selected object."""

    APPROACH_HEIGHT = 0.15
    LIFT_HEIGHT = 0.20

    def __init__(self, simulation: FrankaSimulation, on_step: Callable[[str], None] | None = None) -> None:
        self.sim = simulation
        self.model = simulation.model
        self._ik_data = mujoco.MjData(self.model)
        self._lower = self.model.jnt_range[:ARM_JOINTS, 0]
        self._upper = self.model.jnt_range[:ARM_JOINTS, 1]
        self._phase = "idle"
        self._on_step = on_step
        self._should_abort: Callable[[], bool] | None = None

    def plan_grasp(self, target: str) -> np.ndarray:
        """Arm joint angles at the grasp pose for ``target``, without moving the robot."""
        spec = self.sim.object_specs[target]
        position = self.sim.position(target)
        rotation = self._grasp_rotation(np.asarray(spec.grasp_axis, dtype=float))
        grasp = np.array([position[0], position[1], spec.grasp_height])
        q = self._solve_ik(grasp + [0, 0, self.APPROACH_HEIGHT], rotation, self.sim.data.qpos[:ARM_JOINTS].copy(), 200)
        return self._solve_ik(grasp, rotation, q, 200)

    def go_home(self, seconds: float = 2.0) -> None:
        """Smooth joint-space return to the home pose with the gripper open."""
        self._phase = "home"
        start = self.sim.data.qpos[:ARM_JOINTS].copy()
        steps = max(1, int(seconds / self.model.opt.timestep))
        for step in range(steps):
            s = _smoothstep((step + 1) / steps)
            self.sim.data.ctrl[:ARM_JOINTS] = start + s * (self.sim.home_qpos - start)
            self.sim.data.ctrl[ARM_JOINTS] = GRIPPER_OPEN
            self._step()

    def execute(self, action: str, target: str, should_abort: Callable[[], bool] | None = None) -> dict[str, object]:
        """Pick ``target``. ``should_abort`` is polled every physics step until the gripper closes."""
        self._should_abort = should_abort
        try:
            return self._pick(action, target)
        except PickAborted as aborted:
            return {"action": action, "target": target, "aborted": True, "aborted_during": str(aborted),
                    "lifted": False, "grasp_success": False}
        finally:
            self._should_abort = None

    def _pick(self, action: str, target: str) -> dict[str, object]:
        if action != "pick":
            raise ValueError(f"Unsupported action: {action!r}")
        if target not in self.sim.OBJECTS:
            raise ValueError(f"Unknown target object: {target!r}")
        spec = self.sim.object_specs[target]
        others = {name: self.sim.position(name) for name in self.sim.OBJECTS if name != target}
        start = self.sim.position(target)
        rotation = self._grasp_rotation(np.asarray(spec.grasp_axis, dtype=float))
        grasp = np.array([start[0], start[1], spec.grasp_height])
        above = grasp + [0, 0, self.APPROACH_HEIGHT]

        self._move("approach", above, rotation, seconds=2.0, gripper=GRIPPER_OPEN)
        self._move("descend", grasp, rotation, seconds=1.2, gripper=GRIPPER_OPEN)
        self._hold("grasp", seconds=0.8, gripper=GRIPPER_CLOSED)
        contacts_at_grasp = self.sim.finger_contacts(target)
        self._move("lift", grasp + [0, 0, self.LIFT_HEIGHT], rotation, seconds=1.8, gripper=GRIPPER_CLOSED)
        self._hold("hold", seconds=1.0, gripper=GRIPPER_CLOSED)

        final = self.sim.position(target)
        tcp = self.sim.tcp_position()
        disturbed = max(float(np.linalg.norm(self.sim.position(name)[:2] - xy[:2])) for name, xy in others.items())
        lifted = bool(final[2] > start[2] + 0.10)
        in_hand = bool(np.linalg.norm(final - tcp) < 0.04)
        both_fingers = self.sim.finger_contacts(target) == {"left", "right"}
        return {
            "action": action,
            "target": target,
            "initial_z": float(start[2]),
            "final_z": float(final[2]),
            "lift_delta": float(final[2] - start[2]),
            "grasp_success": contacts_at_grasp == {"left", "right"},
            "fingers_in_contact_after_lift": both_fingers,
            "object_tcp_distance": float(np.linalg.norm(final - tcp)),
            "max_other_object_shift": disturbed,
            "lifted": lifted and in_hand and both_fingers,
            "aborted": False,
        }

    def _grasp_rotation(self, closing_axis: np.ndarray) -> np.ndarray:
        """Gripper pointing straight down with fingers closing along ``closing_axis``."""
        z = np.array([0.0, 0.0, -1.0])
        y = np.array([closing_axis[0], closing_axis[1], 0.0])
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        return np.column_stack((x, y, z))

    def _move(self, phase: str, position: np.ndarray, rotation: np.ndarray, seconds: float, gripper: float) -> None:
        """Straight-line Cartesian motion with smooth timing, tracked by joint position control."""
        self._phase = phase
        start_position = self.sim.tcp_position()
        start_rotation = self.sim.data.site_xmat[self.sim.tcp_site].reshape(3, 3).copy()
        q = self.sim.data.qpos[:ARM_JOINTS].copy()
        steps = max(1, int(seconds / self.model.opt.timestep))
        waypoints = max(2, int(seconds * 25))
        targets = []
        for index in range(1, waypoints + 1):
            s = _smoothstep(index / waypoints)
            q = self._solve_ik(start_position + s * (position - start_position), _slerp(start_rotation, rotation, s), q)
            targets.append(q)
        targets = np.asarray(targets)
        previous = self.sim.data.qpos[:ARM_JOINTS].copy()
        for step in range(steps):
            u = (step + 1) / steps * waypoints
            lower = int(np.floor(u)) - 1
            frac = u - np.floor(u)
            a = previous if lower < 0 else targets[min(lower, waypoints - 1)]
            b = targets[min(lower + 1, waypoints - 1)]
            self.sim.data.ctrl[:ARM_JOINTS] = a + frac * (b - a)
            self.sim.data.ctrl[ARM_JOINTS] = gripper
            self._step()

    def _hold(self, phase: str, seconds: float, gripper: float) -> None:
        self._phase = phase
        self.sim.data.ctrl[ARM_JOINTS] = gripper
        for _ in range(int(seconds / self.model.opt.timestep)):
            self._step()

    def _step(self) -> None:
        self.sim.step()
        if self._on_step is not None:
            self._on_step(self._phase)
        if self._should_abort is not None and self._phase in ABORTABLE_PHASES and self._should_abort():
            raise PickAborted(self._phase)

    def _solve_ik(self, position: np.ndarray, rotation: np.ndarray, q_init: np.ndarray, iterations: int = 60) -> np.ndarray:
        """Damped least-squares IK for the tool-centre-point site (position and orientation)."""
        data = self._ik_data
        data.qpos[:] = self.sim.data.qpos
        q = q_init.copy()
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, rotation.flatten())
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        site_quat = np.zeros(4)
        rotation_error = np.zeros(3)
        for _ in range(iterations):
            data.qpos[:ARM_JOINTS] = q
            mujoco.mj_kinematics(self.model, data)
            mujoco.mj_comPos(self.model, data)
            position_error = position - data.site_xpos[self.sim.tcp_site]
            mujoco.mju_mat2Quat(site_quat, data.site_xmat[self.sim.tcp_site])
            mujoco.mju_subQuat(rotation_error, target_quat, site_quat)
            # mju_subQuat returns the error in the site frame; the Jacobian is in the world frame.
            rotation_error = data.site_xmat[self.sim.tcp_site].reshape(3, 3) @ rotation_error
            error = np.concatenate((position_error, rotation_error))
            if np.linalg.norm(position_error) < 1e-4 and np.linalg.norm(rotation_error) < 1e-3:
                break
            mujoco.mj_jacSite(self.model, data, jacp, jacr, self.sim.tcp_site)
            jacobian = np.vstack((jacp, jacr))[:, :ARM_JOINTS]
            damping = 1e-3
            dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(6), error)
            norm = np.linalg.norm(dq)
            if norm > 0.2:
                dq *= 0.2 / norm
            q = np.clip(q + dq, self._lower, self._upper)
        return q


def _smoothstep(s: float) -> float:
    return s * s * (3 - 2 * s)


def _slerp(start: np.ndarray, end: np.ndarray, s: float) -> np.ndarray:
    q0, q1 = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q0, start.flatten())
    mujoco.mju_mat2Quat(q1, end.flatten())
    if np.dot(q0, q1) < 0:
        q1 = -q1
    angle = np.arccos(np.clip(np.dot(q0, q1), -1, 1))
    q = q0 if angle < 1e-6 else (np.sin((1 - s) * angle) * q0 + np.sin(s * angle) * q1) / np.sin(angle)
    q /= np.linalg.norm(q)
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, q)
    return matrix.reshape(3, 3)
