"""Franka Emika Panda tabletop scene in MuJoCo with physically grasped objects.

The robot model is MuJoCo Menagerie's ``franka_emika_panda`` (fetched by
``scripts/fetch_franka.sh``) loaded unmodified; the table, objects, lights,
cameras, and a tool-centre-point site are added through ``MjSpec``. Objects
are free bodies: they move only through contact and gravity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


PANDA_XML = Path(__file__).resolve().parents[1] / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"
ARM_JOINTS = 7
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0
TCP_OFFSET = 0.1034  # hand origin to the centre of the fingertip pads (m)


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    xy: tuple[float, float]
    geom: str
    size: tuple[float, ...]
    rgba: tuple[float, float, float, float]
    grasp_axis: tuple[float, float]  # world direction the fingers close along
    grasp_height: float  # tool-centre-point height above the table when grasping
    mass: float
    friction: float = 1.5  # sliding friction against the rubber fingertip pads


OBJECTS = (
    ObjectSpec("APPLE", (0.50, -0.22), "sphere", (0.032,), (0.80, 0.08, 0.06, 1), (0.0, 1.0), 0.032, 0.12, 2.0),
    ObjectSpec("CUP", (0.55, 0.0), "cylinder", (0.030, 0.045), (0.92, 0.92, 0.95, 1), (0.0, 1.0), 0.050, 0.10),
    ObjectSpec("PHONE", (0.50, 0.22), "box", (0.028, 0.065, 0.012), (0.08, 0.09, 0.12, 1), (1.0, 0.0), 0.016, 0.15),
)


class FrankaSimulation:
    """Panda arm, table, and the selectable objects; nothing is moved kinematically."""

    OBJECTS = tuple(spec.name for spec in OBJECTS)

    def __init__(self, gui: bool = False) -> None:
        if not PANDA_XML.is_file():
            raise FileNotFoundError(f"Franka model not found at {PANDA_XML}; run scripts/fetch_franka.sh")
        self.object_specs = {spec.name: spec for spec in OBJECTS}
        self.model = _build_model()
        self.data = mujoco.MjData(self.model)
        self.tcp_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.left_finger = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self.object_bodies = {name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.OBJECTS}
        self.viewer = None
        self.reset()
        if gui:
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)

    def reset(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[:ARM_JOINTS] = self.data.qpos[:ARM_JOINTS]
        self.data.ctrl[ARM_JOINTS] = GRIPPER_OPEN
        mujoco.mj_forward(self.model, self.data)
        self.home_qpos = self.data.qpos[:ARM_JOINTS].copy()
        self.step(300)  # let objects settle on the table

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()

    def position(self, name: str) -> np.ndarray:
        return self.data.xpos[self.object_bodies[name]].copy()

    def tcp_position(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp_site].copy()

    def finger_contacts(self, name: str) -> set[str]:
        """Which fingers currently touch the named object."""
        body = self.object_bodies[name]
        touching = set()
        for contact in self.data.contact[: self.data.ncon]:
            bodies = {self.model.geom_bodyid[contact.geom1], self.model.geom_bodyid[contact.geom2]}
            if body in bodies:
                if self.left_finger in bodies:
                    touching.add("left")
                if self.right_finger in bodies:
                    touching.add("right")
        return touching

    def render(self, renderer: mujoco.Renderer, camera: str = "overview") -> np.ndarray:
        renderer.update_scene(self.data, camera=camera)
        return renderer.render().copy()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def _build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.visual.headlight.diffuse = [0.4, 0.4, 0.4]

    spec.add_texture(name="floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.10, 0.11, 0.13], rgb2=[0.13, 0.14, 0.17], width=512, height=512)
    spec.add_material(name="floor_mat", textures=["", "floor_tex"], texrepeat=[8, 8], reflectance=0.15)
    spec.add_material(name="table_mat", rgba=[0.32, 0.22, 0.15, 1], reflectance=0.08)

    world = spec.worldbody
    world.add_light(name="key", pos=[0.6, -0.8, 1.8], dir=[-0.3, 0.45, -1], castshadow=True,
                    diffuse=[0.75, 0.75, 0.75])
    world.add_light(name="fill", pos=[-0.6, 0.8, 1.6], dir=[0.4, -0.4, -1], castshadow=False, diffuse=[0.3, 0.3, 0.35])
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, -0.76], material="floor_mat")
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.75, 0.6, 0.025], pos=[0.35, 0, -0.025],
                   material="table_mat", friction=[1.0, 0.005, 0.0001])
    for x in (-0.35, 1.05):
        for y in (-0.55, 0.55):
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, 0.355], pos=[x, y, -0.405],
                           material="table_mat", contype=0, conaffinity=0)

    spec.body("hand").add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.005, 0.005, 0.005], rgba=[0, 1, 0, 0])

    for item in OBJECTS:
        height = item.size[{"sphere": 0, "cylinder": 1, "box": 2}[item.geom]]
        body = world.add_body(name=item.name, pos=[item.xy[0], item.xy[1], height + 0.002])
        body.add_freejoint()
        body.add_geom(name=f"{item.name}_geom", type=getattr(mujoco.mjtGeom, f"mjGEOM_{item.geom.upper()}"),
                      size=list(item.size) + [0] * (3 - len(item.size)), rgba=list(item.rgba), mass=item.mass,
                      friction=[item.friction, 0.02, 0.001], condim=4, solref=[0.01, 1])
        _decorate(body, item)

    world.add_camera(name="overview", pos=[1.30, -0.90, 0.80], xyaxes=_look_at([1.30, -0.90, 0.80], [0.40, 0.0, 0.20]))
    world.add_camera(name="robot_view", pos=[1.05, 0, 0.75], xyaxes=_look_at([1.05, 0, 0.75], [0.50, 0.0, 0.0]))
    return spec.compile()


def _look_at(eye, target) -> list[float]:
    """Camera ``xyaxes`` that point the camera from ``eye`` at ``target`` with world z up."""
    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return [*right, *up]


def _decorate(body, item: ObjectSpec) -> None:
    """Visual-only details; they never collide."""
    visual = dict(contype=0, conaffinity=0, mass=0)
    if item.name == "APPLE":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.003, 0.008, 0], pos=[0, 0, 0.036],
                      rgba=[0.35, 0.22, 0.1, 1], **visual)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.012, 0.005, 0.002], pos=[0.009, 0, 0.038],
                      euler=[0, 0.4, 0], rgba=[0.2, 0.6, 0.15, 1], **visual)
    elif item.name == "CUP":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.026, 0.002, 0], pos=[0, 0, 0.044],
                      rgba=[0.35, 0.2, 0.1, 1], **visual)  # coffee surface
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.005, 0.018, 0], pos=[0.038, 0, 0.0],
                      rgba=list(item.rgba), **visual)  # handle
    elif item.name == "PHONE":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.060, 0.0005], pos=[0, 0, 0.0122],
                      rgba=[0.15, 0.45, 0.95, 1], **visual)  # screen
