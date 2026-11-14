"""Three-panel demo frame: what the subject sees, the brain signal, and the robot."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402

from simulation.franka_environment import ARM_JOINTS, FrankaSimulation  # noqa: E402
from .look_track import LookTrack, spectrum  # noqa: E402

BG = "#0d1117"
PANEL = "#161b22"
TEXT = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#7ee787"
COLORS = {"APPLE": "#ff6b6b", "CUP": "#f2cc60", "PHONE": "#58a6ff", "CANCEL": "#d2a8ff"}
FLICKER_SLOWDOWN = 8.0  # rings pulse 8x slower than the real stimulus (photosensitivity-safe)
VIEW_SIZE = (600, 640)  # height, width of each MuJoCo panel


@dataclass
class DemoState:
    """Everything a frame shows. The driver mutates this; the renderer only reads it."""

    time: float = 0.0
    subject: int = 1
    look: LookTrack | None = None
    look_elapsed: float = 0.0
    last_look: LookTrack | None = None
    candidate: str | None = None
    confirmed: str | None = None
    ghost_qpos: np.ndarray | None = None
    robot_phase: str = "idle"
    banner: str = ""
    banner_color: str = TEXT
    history: list[str] = field(default_factory=list)
    stats: dict[str, str] = field(default_factory=dict)
    card: tuple[str, str, str] | None = None  # (title, subtitle, footnote) overlay
    card_color: str = ACCENT


class HeroRenderer:
    def __init__(self, simulation: FrankaSimulation, targets: dict[str, float], channel_names: list[str],
                 eeg_info, threshold: float, width: int = 1920, height: int = 1080) -> None:
        self.sim = simulation
        self.targets = targets
        self.names = list(targets)
        self.channel_names = channel_names
        self.eeg_info = eeg_info
        self.threshold = threshold
        self.width, self.height = width, height
        self.view_renderer = mujoco.Renderer(simulation.model, *VIEW_SIZE)
        self.robot_renderer = mujoco.Renderer(simulation.model, *VIEW_SIZE)
        self._ghost_data = mujoco.MjData(simulation.model)
        self._option = mujoco.MjvOption()
        self._perturb = mujoco.MjvPerturb()
        self._robot_bodies = {i for i in range(simulation.model.nbody)
                              if i not in simulation.object_bodies.values() and i != 0}
        self._topomap_cache: dict[tuple[int, float], np.ndarray] = {}
        self.fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=BG)

    # ------------------------------------------------------------------ frame
    def frame(self, state: DemoState) -> np.ndarray:
        fig = self.fig
        fig.clf()
        fig.set_facecolor(BG)
        fig.text(0.012, 0.962, "NEUROPILOT", color=ACCENT, fontsize=26, fontweight="bold", va="center")
        fig.text(0.168, 0.962, "look at an object  →  the robot picks it up", color=TEXT, fontsize=17, va="center")
        fig.text(0.988, 0.962, f"real recorded EEG, replayed  ·  BETA SSVEP subject S{state.subject}  ·  "
                 "training-free CCA  ·  Franka Panda, MuJoCo physics", color=MUTED, fontsize=11, ha="right", va="center")

        self._subject_view(fig.add_axes([0.008, 0.335, 0.325, 0.585]), state)
        self._eeg(fig.add_axes([0.355, 0.665, 0.29, 0.235]), state)
        self._spectrum(fig.add_axes([0.355, 0.365, 0.175, 0.235]), state)
        self._topomap(fig.add_axes([0.545, 0.345, 0.11, 0.27]), state)
        self._robot_view(fig.add_axes([0.667, 0.335, 0.325, 0.585]), state)
        self._score_race(fig.add_axes([0.065, 0.05, 0.255, 0.235]), state)
        self._decisions(fig.add_axes([0.355, 0.03, 0.29, 0.27]), state)
        self._system(fig.add_axes([0.667, 0.03, 0.325, 0.27]), state)
        if state.banner and state.card is None:
            fig.text(0.5, 0.315, state.banner, color=state.banner_color, fontsize=19, fontweight="bold",
                     ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.45", facecolor=PANEL, edgecolor=state.banner_color, linewidth=2))
        if state.card is not None:
            self._card(state.card, state.card_color)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    # ------------------------------------------------------------ left panel
    def _subject_view(self, ax, state: DemoState) -> None:
        self.view_renderer.update_scene(self.sim.data, camera="robot_view")
        ax.imshow(self.view_renderer.render())
        _panel_title(ax, "WHAT THE SUBJECT SEES", "each object flickers at its own rate · shown 8× slower")
        ax.set_axis_off()
        camera = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "robot_view")
        height, width = VIEW_SIZE
        for name in self.sim.OBJECTS:
            u, v = _project(self.sim, camera, self.sim.position(name), width, height)
            self._tag(ax, name, u, v, 46, state)
        self._tag(ax, "CANCEL", width - 70, height - 95, 40, state, square=True)

    def _tag(self, ax, name: str, u: float, v: float, radius: float, state: DemoState, square: bool = False) -> None:
        color = COLORS[name]
        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * self.targets[name] / FLICKER_SLOWDOWN * state.time)
        alpha = 0.25 + 0.55 * pulse
        if square:
            ax.add_patch(Rectangle((u - radius, v - radius), 2 * radius, 2 * radius, facecolor=color, alpha=alpha * 0.6,
                                   edgecolor=color, linewidth=2))
            ax.text(u, v, "✕", color="white", fontsize=22, ha="center", va="center", fontweight="bold")
        else:
            ax.add_patch(Circle((u, v), radius, fill=False, edgecolor=color, linewidth=3 + 3 * pulse, alpha=alpha))
        ax.text(u, v + radius + 16, f"{name}  {self.targets[name]:.1f} Hz", color=color, fontsize=10,
                ha="center", va="center", fontweight="bold")
        if name == state.confirmed:
            _reticle(ax, u, v, radius + 14, ACCENT, "CONFIRMED")
        elif name == state.candidate:
            _reticle(ax, u, v, radius + 14, color, "PREVIEW · look again")

    # ---------------------------------------------------------- middle panel
    def _eeg(self, ax, state: DemoState) -> None:
        track, elapsed, live = self._active_track(state)
        _style(ax)
        _panel_title(ax, "BRAIN · occipital EEG", "looking at:  " + (track.intended if track else "—")
                     + ("   (dataset label)  ·  seconds into the look" if track else ""))
        ax.set_xlim(0, 2.0)
        ax.set_yticks([])
        if track is None:
            return
        samples = int(round(elapsed * track.sampling_rate_hz))
        time = np.arange(samples) / track.sampling_rate_hz
        scale = 4 * np.median(np.abs(track.eeg)) + 1e-12
        for row, name in enumerate(self.channel_names):
            offset = len(self.channel_names) - row
            ax.plot(time, track.eeg[row, :samples] / scale + offset, color=ACCENT if live else MUTED, linewidth=0.8,
                    alpha=0.95 if live else 0.45)
            ax.text(-0.03, offset, name, color=MUTED, fontsize=8, ha="right", va="center")
        ax.set_ylim(0, len(self.channel_names) + 1)

    def _spectrum(self, ax, state: DemoState) -> None:
        track, elapsed, live = self._active_track(state)
        _style(ax)
        _panel_title(ax, "frequency spectrum", "O1 · Oz · O2 mean, current window")
        ax.set_xlim(6, 34)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([])
        ax.set_xlabel("Hz", color=MUTED, fontsize=9)
        for name, frequency in self.targets.items():
            ax.axvline(frequency, color=COLORS[name], linestyle=":", linewidth=1.2, alpha=0.8)
            ax.axvline(2 * frequency, color=COLORS[name], linestyle=":", linewidth=0.8, alpha=0.35)
        if track is None or elapsed < 0.25:
            return
        samples = int(round(elapsed * track.sampling_rate_hz))
        occipital = [self.channel_names.index(c) for c in ("O1", "Oz", "O2")]
        freqs, magnitude = spectrum(track.eeg[occipital, :samples], track.sampling_rate_hz)
        band = (freqs >= 6) & (freqs <= 34)
        magnitude = magnitude / (magnitude[band].max() + 1e-12)
        ax.fill_between(freqs[band], magnitude[band], color=ACCENT if live else MUTED, alpha=0.35)
        ax.plot(freqs[band], magnitude[band], color=ACCENT if live else MUTED, linewidth=1.3)

    def _topomap(self, ax, state: DemoState) -> None:
        track, elapsed, _ = self._active_track(state)
        ax.set_axis_off()
        result = track.topomap_at(elapsed) if track else None
        if result is None:
            ax.text(0.5, 0.5, "response map\nafter 0.5 s", color=MUTED, fontsize=10, ha="center", va="center",
                    transform=ax.transAxes)
            return
        checkpoint, leader, values = result
        key = (id(track), checkpoint)
        if key not in self._topomap_cache:
            self._topomap_cache[key] = _render_topomap(values, self.eeg_info)
        ax.imshow(self._topomap_cache[key])
        ax.set_title(f"response at {self.targets[leader]:.1f} Hz\n({checkpoint:.1f} s)", color=COLORS[leader],
                     fontsize=10, pad=2)

    # ----------------------------------------------------------- right panel
    def _robot_view(self, ax, state: DemoState) -> None:
        renderer = self.robot_renderer
        renderer.update_scene(self.sim.data, camera="overview")
        if state.ghost_qpos is not None:
            self._add_ghost(renderer.scene, state.ghost_qpos)
        ax.imshow(renderer.render())
        ax.set_axis_off()
        phase = {"idle": "waiting for a confirmed selection", "approach": "reaching", "descend": "descending",
                 "grasp": "closing gripper", "lift": "lifting", "hold": "holding", "home": "returning home"}
        _panel_title(ax, "ROBOT · Franka Panda", phase.get(state.robot_phase, state.robot_phase)
                     + ("  ·  ghost = planned grasp" if state.ghost_qpos is not None else ""))

    def _add_ghost(self, scene, qpos: np.ndarray) -> None:
        data = self._ghost_data
        data.qpos[:] = self.sim.data.qpos
        data.qpos[:ARM_JOINTS] = qpos
        mujoco.mj_kinematics(self.sim.model, data)
        start = scene.ngeom
        mujoco.mjv_addGeoms(self.sim.model, data, self._option, self._perturb, mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
        for index in range(start, scene.ngeom):
            geom = scene.geoms[index]
            body = self.sim.model.geom_bodyid[geom.objid] if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM else -1
            geom.rgba[:] = (0.35, 0.85, 1.0, 0.28) if body in self._robot_bodies else (0, 0, 0, 0)

    # ---------------------------------------------------------- bottom band
    def _score_race(self, ax, state: DemoState) -> None:
        track, elapsed, live = self._active_track(state)
        _style(ax)
        window = f"window {elapsed:.2f} s" if track else "waiting for a look"
        _panel_title(ax, "DECODER · CCA score per target", f"{window}  ·  decision at 2.0 s, UNSURE if margin < {self.threshold:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, len(self.names) - 0.4)
        ax.set_yticks(range(len(self.names)), self.names[::-1], color=TEXT, fontsize=11)
        ax.tick_params(axis="x", colors=MUTED, labelsize=8)
        scores = track.scores_at(elapsed) if track is not None else np.full(len(self.names), np.nan)
        leader = int(np.nanargmax(scores)) if np.isfinite(scores).any() else None
        for row, name in enumerate(self.names):
            y = len(self.names) - 1 - row
            value = 0 if not np.isfinite(scores[row]) else scores[row]
            ax.barh(y, 1, color=PANEL, height=0.62)
            ax.barh(y, value, color=COLORS[name], alpha=0.95 if row == leader and live else 0.45, height=0.62)
            if np.isfinite(scores[row]):
                ax.text(min(value + 0.015, 0.93), y, f"{value:.2f}", color=TEXT, fontsize=9, va="center")

    def _decisions(self, ax, state: DemoState) -> None:
        ax.set_axis_off()
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02", facecolor=PANEL,
                                    edgecolor="#30363d", transform=ax.transAxes))
        ax.text(0.03, 0.9, "DECISIONS", color=TEXT, fontsize=12, fontweight="bold", transform=ax.transAxes)
        for row, line in enumerate(state.history[-7:]):
            ax.text(0.03, 0.76 - row * 0.105, line, color=TEXT if row == len(state.history[-7:]) - 1 else MUTED,
                    fontsize=10, family="monospace", transform=ax.transAxes)

    def _system(self, ax, state: DemoState) -> None:
        ax.set_axis_off()
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02", facecolor=PANEL,
                                    edgecolor="#30363d", transform=ax.transAxes))
        ax.text(0.03, 0.9, "MEASURED  (held-out subjects S1–S10)", color=TEXT, fontsize=12, fontweight="bold",
                transform=ax.transAxes)
        for row, (key, value) in enumerate(state.stats.items()):
            ax.text(0.03, 0.75 - row * 0.115, key, color=MUTED, fontsize=10.5, transform=ax.transAxes)
            ax.text(0.97, 0.75 - row * 0.115, value, color=TEXT, fontsize=10.5, ha="right", fontweight="bold",
                    transform=ax.transAxes)

    def _card(self, card: tuple[str, str, str], color: str) -> None:
        title, subtitle, footnote = card
        overlay = self.fig.add_axes([0, 0, 1, 1])
        overlay.set_axis_off()
        overlay.add_patch(Rectangle((0, 0), 1, 1, color=BG, alpha=0.88, transform=overlay.transAxes))
        overlay.text(0.5, 0.58, title, color=color, fontsize=46, fontweight="bold", ha="center", va="center")
        overlay.text(0.5, 0.47, subtitle, color=TEXT, fontsize=21, ha="center", va="center")
        overlay.text(0.5, 0.36, footnote, color=MUTED, fontsize=13, ha="center", va="center", linespacing=1.6)

    # --------------------------------------------------------------- helpers
    def _active_track(self, state: DemoState) -> tuple[LookTrack | None, float, bool]:
        if state.look is not None:
            return state.look, state.look_elapsed, True
        if state.last_look is not None:
            return state.last_look, 2.0, False
        return None, 0.0, False

    def close(self) -> None:
        self.view_renderer.close()
        self.robot_renderer.close()
        plt.close(self.fig)


def _project(sim: FrankaSimulation, camera: int, point: np.ndarray, width: int, height: int) -> tuple[float, float]:
    """Pixel coordinates of a world point in a MuJoCo camera image."""
    position = sim.data.cam_xpos[camera]
    rotation = sim.data.cam_xmat[camera].reshape(3, 3)
    local = rotation.T @ (point - position)
    focal = (height / 2) / np.tan(np.deg2rad(sim.model.cam_fovy[camera]) / 2)
    return width / 2 + focal * local[0] / -local[2], height / 2 - focal * local[1] / -local[2]


def _reticle(ax, u: float, v: float, radius: float, color: str, label: str) -> None:
    for angle in (45, 135, 225, 315):
        a = np.deg2rad(angle)
        inner, outer = radius * 0.75, radius * 1.15
        ax.plot([u + inner * np.cos(a), u + outer * np.cos(a)], [v + inner * np.sin(a), v + outer * np.sin(a)],
                color=color, linewidth=3)
    ax.text(u, v - radius - 14, label, color=color, fontsize=11, ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=BG, edgecolor=color, alpha=0.85))


def _panel_title(ax, title: str, subtitle: str) -> None:
    ax.set_title(f"{title}\n", color=TEXT, fontsize=13, fontweight="bold", loc="left", pad=4)
    ax.text(0, 1.012, subtitle, color=MUTED, fontsize=9.5, transform=ax.transAxes, va="bottom")


def _style(ax) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors=MUTED, labelsize=8)


def _render_topomap(values: np.ndarray, info) -> np.ndarray:
    import mne

    fig, ax = plt.subplots(figsize=(2.4, 2.4), dpi=100, facecolor=BG)
    mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap="magma", contours=0, sensors=False,
                         outlines="head", extrapolate="head")
    for line in ax.lines:
        line.set_color(MUTED)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image
