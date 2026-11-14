"""Three-panel frame for the intracortical demo: monkey's screen, spikes, robot."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Wedge  # noqa: E402

from simulation.franka_environment import ARM_JOINTS, FrankaSimulation  # noqa: E402
from .hero_renderer import ACCENT, BG, COLORS, MUTED, PANEL, TEXT, VIEW_SIZE, _panel_title, _style  # noqa: E402

SECTORS = {"APPLE": (120, 240), "CUP": (60, 120), "PHONE": (0, 60), "CANCEL": (-60, 0)}


@dataclass
class ReachPlayback:
    """One recorded val-split trial played back around movement onset."""

    trial: int
    truth: str
    target_xy: np.ndarray
    spikes: list[np.ndarray]  # per unit (display order), seconds relative to movement onset
    go_cue: float  # relative to movement onset
    target_on: float
    probabilities: list[tuple[float, dict[str, float]]]  # (window end, class probabilities)
    final: dict
    start: float = -0.8
    stop: float = 0.4

    def probabilities_at(self, t: float) -> dict[str, float] | None:
        available = [p for end, p in self.probabilities if end <= t + 1e-9]
        return available[-1] if available else None


@dataclass
class IntracorticalState:
    time: float = 0.0
    playback: ReachPlayback | None = None
    playback_t: float = -0.8
    last: ReachPlayback | None = None
    selected: str | None = None
    ghost_qpos: np.ndarray | None = None
    robot_phase: str = "idle"
    banner: str = ""
    banner_color: str = TEXT
    history: list[str] = field(default_factory=list)
    stats: dict[str, str] = field(default_factory=dict)
    card: tuple[str, str, str] | None = None
    card_color: str = ACCENT


class IntracorticalRenderer:
    def __init__(self, simulation: FrankaSimulation, trials, all_targets: np.ndarray, slowdown: float,
                 width: int = 1920, height: int = 1080) -> None:
        self.sim = simulation
        self.trials = trials
        self.all_targets = all_targets
        self.slowdown = slowdown
        self.width, self.height = width, height
        self.robot_renderer = mujoco.Renderer(simulation.model, *VIEW_SIZE)
        self._ghost_data = mujoco.MjData(simulation.model)
        self._option = mujoco.MjvOption()
        self._perturb = mujoco.MjvPerturb()
        self._robot_bodies = {i for i in range(simulation.model.nbody)
                              if i not in simulation.object_bodies.values() and i != 0}
        self.fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=BG)

    def frame(self, state: IntracorticalState) -> np.ndarray:
        fig = self.fig
        fig.clf()
        fig.set_facecolor(BG)
        fig.text(0.012, 0.962, "NEUROPILOT", color=ACCENT, fontsize=26, fontweight="bold", va="center")
        fig.text(0.168, 0.962, "intracortical intention  →  the robot acts", color=TEXT, fontsize=17, va="center")
        fig.text(0.988, 0.962, "recorded macaque M1/PMd spikes (NLB MC_Maze, monkey J), replayed  ·  "
                 "linear decoder  ·  Franka Panda, MuJoCo physics", color=MUTED, fontsize=11, ha="right", va="center")
        self._screen(fig.add_axes([0.03, 0.36, 0.28, 0.54]), state)
        self._raster(fig.add_axes([0.355, 0.40, 0.29, 0.50]), state)
        self._race(fig.add_axes([0.40, 0.04, 0.245, 0.19]), state)
        self._robot(fig.add_axes([0.667, 0.335, 0.325, 0.585]), state)
        self._log(fig.add_axes([0.012, 0.03, 0.335, 0.27]), state)
        self._stats(fig.add_axes([0.667, 0.03, 0.325, 0.27]), state)
        if state.banner and state.card is None:
            fig.text(0.5, 0.335, state.banner, color=state.banner_color, fontsize=19, fontweight="bold",
                     ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.45", facecolor=PANEL, edgecolor=state.banner_color, linewidth=2))
        if state.card is not None:
            overlay = fig.add_axes([0, 0, 1, 1])
            overlay.set_axis_off()
            overlay.add_patch(Rectangle((0, 0), 1, 1, color=BG, alpha=0.88, transform=overlay.transAxes))
            title, subtitle, footnote = state.card
            overlay.text(0.5, 0.58, title, color=state.card_color, fontsize=44, fontweight="bold", ha="center", va="center")
            overlay.text(0.5, 0.47, subtitle, color=TEXT, fontsize=20, ha="center", va="center")
            overlay.text(0.5, 0.36, footnote, color=MUTED, fontsize=13, ha="center", va="center", linespacing=1.6)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    def _active(self, state):
        if state.playback is not None:
            return state.playback, state.playback_t, True
        if state.last is not None:
            return state.last, state.last.stop, False
        return None, None, False

    def _screen(self, ax, state) -> None:
        play, t, live = self._active(state)
        ax.set_facecolor("#05070a")
        ax.set_xlim(-165, 165)
        ax.set_ylim(-125, 125)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        _panel_title(ax, "WHAT THE MONKEY SEES", "reach task screen · direction sector → robot object")
        for name, (a, b) in SECTORS.items():
            ax.add_patch(Wedge((0, 0), 158, a, b, facecolor=COLORS[name], alpha=0.07, edgecolor=COLORS[name], linewidth=0.6))
            mid = np.deg2rad((a + b) / 2)
            ax.text(140 * np.cos(mid), 108 * np.sin(mid) if name != "APPLE" else -100, name, color=COLORS[name],
                    fontsize=10, ha="center", va="center", fontweight="bold", alpha=0.85)
        for xy in self.all_targets:
            ax.add_patch(Rectangle((xy[0] - 6, xy[1] - 6), 12, 12, facecolor="#30363d", edgecolor="#484f58"))
        ax.add_patch(Circle((0, 0), 5, facecolor="#484f58"))
        if play is None:
            return
        shown = t >= play.target_on
        if shown:
            x, y = play.target_xy
            ax.add_patch(Rectangle((x - 8, y - 8), 16, 16, facecolor=COLORS[play.truth], edgecolor="white", linewidth=1.5))
        cursor = self.trials.cursor_at(play.trial, t)
        ax.add_patch(Circle(cursor, 5.5, facecolor="white", edgecolor=ACCENT, linewidth=1.5))
        phase = ("target shown, waiting" if t < play.go_cue else "go cue, not moving yet" if t < -0.02
                 else "movement onset" if t < 0.03 else "hand moving")
        ax.text(-160, -118, f"trial {play.trial}  ·  t = {t:+.2f} s from movement onset  ·  {phase}",
                color=MUTED, fontsize=9)

    def _raster(self, ax, state) -> None:
        play, t, live = self._active(state)
        _style(ax)
        _panel_title(ax, "BRAIN · 162 neurons, motor & premotor cortex",
                     f"spike raster, sorted by preferred direction · shown {self.slowdown:g}× slower")
        ax.set_yticks([])
        if play is None:
            ax.set_xlim(-0.8, 0.4)
            return
        ax.set_xlim(play.start, play.stop)
        ax.set_ylim(-1, len(play.spikes))
        xs, ys = [], []
        for row, times in enumerate(play.spikes):
            visible = times[(times >= play.start) & (times <= t)]
            xs.append(visible)
            ys.append(np.full(len(visible), row))
        ax.scatter(np.concatenate(xs), np.concatenate(ys), s=14, marker="|", color=ACCENT if live else MUTED,
                   alpha=0.95 if live else 0.55, linewidths=1.4)
        for x, label, color in ((play.target_on, "target on", MUTED), (play.go_cue, "go cue", MUTED),
                                (0.0, "movement onset · decision", "#ffa657")):
            if play.start <= x <= play.stop:
                ax.axvline(x, color=color, linestyle="--", linewidth=1.2)
                ax.text(x + 0.01, len(play.spikes) - 2, label, color=color, fontsize=9, va="top", fontweight="bold")
        ax.axvspan(play.final["window"][0], play.final["window"][1], color="#ffa657", alpha=0.08)
        ax.set_xlabel("seconds relative to movement onset", color=MUTED, fontsize=9)

    def _race(self, ax, state) -> None:
        play, t, live = self._active(state)
        _style(ax)
        names = list(COLORS)
        probabilities = (play.final["scores"] if (play and not live) else play.probabilities_at(t)) if play else None
        subtitle = ("waiting for a trial" if play is None else
                    f"decided at movement onset · LDA posterior {play.final['score']:.2f} (uncalibrated)" if not live else
                    "200 ms window ending now · LDA posterior, uncalibrated" if probabilities else "collecting spikes")
        _panel_title(ax, "DECODER · reach direction probability", subtitle)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.set_yticks(range(len(names)), names[::-1], color=TEXT, fontsize=11)
        values = [probabilities.get(n, 0.0) if probabilities else 0.0 for n in names]
        leader = int(np.argmax(values)) if probabilities else None
        for row, name in enumerate(names):
            y = len(names) - 1 - row
            ax.barh(y, 1, color=PANEL, height=0.62)
            ax.barh(y, values[row], color=COLORS[name], alpha=0.95 if row == leader else 0.45, height=0.62)
            if probabilities:
                ax.text(min(values[row] + 0.015, 0.92), y, f"{values[row]:.2f}", color=TEXT, fontsize=9, va="center")

    def _robot(self, ax, state) -> None:
        renderer = self.robot_renderer
        renderer.update_scene(self.sim.data, camera="overview")
        if state.ghost_qpos is not None:
            data = self._ghost_data
            data.qpos[:] = self.sim.data.qpos
            data.qpos[:ARM_JOINTS] = state.ghost_qpos
            mujoco.mj_kinematics(self.sim.model, data)
            start = renderer.scene.ngeom
            mujoco.mjv_addGeoms(self.sim.model, data, self._option, self._perturb, mujoco.mjtCatBit.mjCAT_DYNAMIC,
                                renderer.scene)
            for index in range(start, renderer.scene.ngeom):
                geom = renderer.scene.geoms[index]
                body = self.sim.model.geom_bodyid[geom.objid] if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM else -1
                geom.rgba[:] = (0.35, 0.85, 1.0, 0.28) if body in self._robot_bodies else (0, 0, 0, 0)
        ax.imshow(renderer.render())
        ax.set_axis_off()
        phase = {"idle": "waiting for a decoded intention", "approach": "reaching", "descend": "descending",
                 "grasp": "closing gripper", "lift": "lifting", "hold": "holding", "home": "returning home"}
        _panel_title(ax, "ROBOT · Franka Panda", phase.get(state.robot_phase, state.robot_phase))

    def _log(self, ax, state) -> None:
        ax.set_axis_off()
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02", facecolor=PANEL,
                                    edgecolor="#30363d", transform=ax.transAxes))
        ax.text(0.03, 0.9, "DECISIONS", color=TEXT, fontsize=12, fontweight="bold", transform=ax.transAxes)
        lines = state.history[-6:]
        for row, line in enumerate(lines):
            ax.text(0.03, 0.76 - row * 0.12, line, color=TEXT if row == len(lines) - 1 else MUTED, fontsize=10,
                    family="monospace", transform=ax.transAxes)

    def _stats(self, ax, state) -> None:
        ax.set_axis_off()
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02", facecolor=PANEL,
                                    edgecolor="#30363d", transform=ax.transAxes))
        ax.text(0.03, 0.9, "MEASURED  (125 held-out trials, official NLB val split)", color=TEXT, fontsize=12,
                fontweight="bold", transform=ax.transAxes)
        for row, (key, value) in enumerate(state.stats.items()):
            ax.text(0.03, 0.75 - row * 0.115, key, color=MUTED, fontsize=10.5, transform=ax.transAxes)
            ax.text(0.97, 0.75 - row * 0.115, value, color=TEXT, fontsize=10.5, ha="right", fontweight="bold",
                    transform=ax.transAxes)

    def close(self) -> None:
        self.robot_renderer.close()
        plt.close(self.fig)
