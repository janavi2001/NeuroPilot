"""Small deterministic MuJoCo tabletop environment for Step 3."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

MODEL_XML = r"""
<mujoco model="neuropilot_step3">
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset>
    <material name="floor_mat" rgba="0.12 0.14 0.18 1"/>
    <material name="table_mat" rgba="0.35 0.22 0.12 1"/>
    <material name="robot_mat" rgba="0.25 0.55 0.95 1"/>
    <material name="apple_mat" rgba="0.85 0.12 0.08 1"/>
    <material name="cup_mat" rgba="0.85 0.85 0.9 1"/>
    <material name="phone_mat" rgba="0.1 0.2 0.65 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="3 3 .1" material="floor_mat"/>
    <geom name="table" type="box" pos=".35 0 .45" size=".7 .5 .05" material="table_mat"/>
    <body name="robot_base" pos="-.35 0 .5"><geom type="cylinder" size=".12 .18" material="robot_mat"/></body>
    <body name="gripper" mocap="true" pos=".32 0 .78">
      <geom name="palm" type="box" pos="0 0 -.045" size=".06 .06 .025" material="robot_mat"/>
      <geom name="finger_left" type="box" pos="-.045 0 -.10" size=".012 .035 .055" material="robot_mat"/>
      <geom name="finger_right" type="box" pos=".045 0 -.10" size=".012 .035 .055" material="robot_mat"/>
    </body>
    <body name="APPLE" pos=".32 0 .555"><freejoint/><geom type="sphere" size=".055" mass=".08" material="apple_mat"/></body>
    <body name="CUP" pos=".48 0 .57"><freejoint/><geom type="cylinder" size=".045 .05" mass=".08" material="cup_mat"/></body>
    <body name="PHONE" pos=".64 0 .562"><freejoint/><geom type="box" size=".045 .075 .012" mass=".08" material="phone_mat"/></body>
  </worldbody>
</mujoco>
"""


@dataclass(frozen=True)
class SimObject:
    name: str
    body_id: int
    initial_z: float


class RobotSimulation:
    """Local CPU simulation with a visible scripted gripper and tabletop objects."""

    OBJECTS = ("APPLE", "CUP", "PHONE")

    def __init__(self, gui: bool = False) -> None:
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        if gui:
            try:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            except Exception as exc:  # pragma: no cover - depends on desktop display
                raise RuntimeError(f"Could not open MuJoCo viewer: {exc}") from exc
        mujoco.mj_forward(self.model, self.data)
        self.objects = {
            name: SimObject(name, mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name), self.position(name)[2])
            for name in self.OBJECTS
        }

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()

    def set_gripper(self, position: tuple[float, float, float]) -> None:
        self.data.mocap_pos[0] = position
        mujoco.mj_forward(self.model, self.data)

    def gripper_position(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self.data.mocap_pos[0])

    def position(self, target: str) -> tuple[float, float, float]:
        if target not in self.OBJECTS:
            raise KeyError(f"Unknown simulation object: {target}")
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target)
        return tuple(float(value) for value in self.data.xpos[body_id])

    def attach_to_gripper(self, target: str) -> None:
        body_id = self.objects[target].body_id
        joint_id = self.model.body_jntadr[body_id]
        qpos_address = self.model.jnt_qposadr[joint_id]
        self.data.qpos[qpos_address : qpos_address + 3] = self.gripper_position()
        self.data.qpos[qpos_address + 3 : qpos_address + 7] = (1, 0, 0, 0)
        mujoco.mj_forward(self.model, self.data)

    def gr00t_observation(self, instruction: str) -> dict[str, object]:
        """Build the DROID/XDOF-shaped observation expected by GR00T N1.7.

        The base model's pretraining contract uses two RGB frames, a 9D end
        effector pose, one gripper value, seven joint values, and language.
        This adapter is explicit so incompatible shapes fail at the policy
        boundary instead of silently falling back to scripted control.
        """
        frame = self.render_rgb()
        position = np.asarray(self.gripper_position(), dtype=np.float32)
        eef_9d = np.concatenate((position, np.asarray((1, 0, 0, 0, 1, 0), dtype=np.float32)))
        return {
            "video": {
                "exterior_image_1_left": np.stack((frame, frame))[None, ...],
                "wrist_image_left": np.stack((frame, frame))[None, ...],
            },
            "state": {
                "eef_9d": eef_9d[None, None, :],
                "gripper_position": np.ones((1, 1, 1), dtype=np.float32),
                "joint_position": np.zeros((1, 1, 7), dtype=np.float32),
            },
            "language": {"annotation.language.language_instruction": [[instruction]]},
        }

    def render_rgb(self) -> np.ndarray:
        renderer = mujoco.Renderer(self.model, height=224, width=224)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (0.35, 0.0, 0.52)
        camera.distance = 1.35
        camera.azimuth = 125
        camera.elevation = -22
        renderer.update_scene(self.data, camera)
        frame = renderer.render().copy()
        renderer.close()
        return frame

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
