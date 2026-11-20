"""End-to-end demo frame: real EEG decision -> GR00T action -> Cosmos worlds, one 1920x1080 frame at a time."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from .hero_renderer import ACCENT, BG, MUTED, PANEL, TEXT, _panel_title, _style  # noqa: E402
from .look_track import LookTrack  # noqa: E402

COLORS = {"BOWL": "#f0883e", "PLATE": "#58a6ff", "CHEESE": "#e3d026", "WINE": "#ff6b6b", "STOVE": "#39c5cf",
          "CANCEL": "#d2a8ff", "UNSURE": MUTED}
STAGES = ("LOOK", "CONFIRM", "INSTRUCTION", "GR00T ACTS", "CHECKED")
WARN = "#f0883e"


@dataclass
class MissionState:
    """Everything one frame shows; the timeline builds these, the renderer only reads them."""

    scene: str = "kitchen"  # title | kitchen | worlds | end
    command: int = 0
    commands_total: int = 5
    intended: str | None = None
    look: LookTrack | None = None
    look_elapsed: float = 0.0
    live: bool = False
    stage: int = 0  # index into STAGES
    candidate: str | None = None
    confirmed: str | None = None
    instruction: str = ""
    banner: str = ""
    banner_color: str = TEXT
    robot_frame: np.ndarray | None = None
    sim_frame: np.ndarray | None = None
    robot_caption: str = ""
    robot_title: str = "ROBOT · NVIDIA GR00T N1.7, pretrained, zero fine-tuning"
    world_label: str = ""  # tag on the robot panel, e.g. "office · NVIDIA Cosmos 3"
    wipe: float | None = None  # simulator -> Cosmos reveal position (0..1); robot_frame is already composited
    explain_frames: list[np.ndarray] = field(default_factory=list)  # simulator, edge control, Cosmos
    history: list[str] = field(default_factory=list)
    stats: dict[str, str] = field(default_factory=dict)
    world_frames: dict[str, np.ndarray] = field(default_factory=dict)
    world_titles: dict[str, str] = field(default_factory=dict)
    card: tuple[str, str, list[str]] | None = None  # title, subtitle, lines
    fade: float = 1.0  # card opacity


class MissionRenderer:
    def __init__(self, targets: dict[str, float], channel_names: list[str], threshold: float,
                 width: int = 1920, height: int = 1080) -> None:
        self.targets = targets
        self.names = list(targets)
        self.channel_names = channel_names
        self.threshold = threshold
        self.fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=BG)

    def frame(self, state: MissionState) -> np.ndarray:
        fig = self.fig
        fig.clf()
        fig.set_facecolor(BG)
        if state.scene == "kitchen":
            self._kitchen(state)
        elif state.scene == "worlds":
            self._worlds(state)
        elif state.scene == "explain":
            self._explain(state)
        if state.card is not None:
            self._card(state)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    # ----------------------------------------------------------------- scenes
    def _header(self, right: str) -> None:
        self.fig.text(0.012, 0.962, "NEUROPILOT", color=ACCENT, fontsize=26, fontweight="bold", va="center")
        self.fig.text(0.168, 0.962, "brain signals choose  →  a robot acts", color=TEXT, fontsize=17, va="center")
        self.fig.text(0.988, 0.962, right, color=MUTED, fontsize=11, ha="right", va="center")

    def _kitchen(self, state: MissionState) -> None:
        self._header("real recorded EEG, replayed  ·  BETA SSVEP, held-out subject S1  ·  "
                     "NVIDIA GR00T N1.7 in LIBERO  ·  NVIDIA Cosmos 3")
        self._eeg(self.fig.add_axes([0.035, 0.62, 0.3, 0.27]), state)
        self._race(self.fig.add_axes([0.075, 0.31, 0.26, 0.24]), state)
        self._history(self.fig.add_axes([0.015, 0.03, 0.32, 0.22]), state)
        self._robot(self.fig.add_axes([0.355, 0.13, 0.395, 0.76]), state)
        if state.sim_frame is not None:
            self._sim(self.fig.add_axes([0.77, 0.57, 0.215, 0.32]), state)
        self._pipeline(self.fig.add_axes([0.77, 0.24, 0.215, 0.27]), state)
        self._stats(self.fig.add_axes([0.77, 0.03, 0.215, 0.18]), state)
        if state.banner:
            self.fig.text(0.5525, 0.075, state.banner, color=state.banner_color, fontsize=18, fontweight="bold",
                          ha="center", va="center",
                          bbox=dict(boxstyle="round,pad=0.45", facecolor=PANEL, edgecolor=state.banner_color,
                                    linewidth=2))

    def _explain(self, state: MissionState) -> None:
        self._header("why NVIDIA Cosmos  ·  the same BOWL run you just saw")
        self.fig.text(0.5, 0.885, "Robots learn in simulators.  Real homes look different.", color=TEXT, fontsize=26,
                      fontweight="bold", ha="center", va="center")
        self.fig.text(0.5, 0.835, "NVIDIA Cosmos 3 keeps the exact geometry and motion of a real run and repaints the "
                      "world around it, so one robot run becomes many realistic homes.", color=MUTED, fontsize=14,
                      ha="center", va="center")
        titles = [("1 · THE REAL RUN", "GR00T in the LIBERO simulator"),
                  ("2 · WHAT COSMOS KEEPS", "edges of the robot, objects and motion"),
                  ("3 · WHAT COSMOS MAKES", "photoreal kitchen, same motion")]
        for index, (image, (title, subtitle)) in enumerate(zip(state.explain_frames, titles)):
            ax = self.fig.add_axes([0.03 + index * 0.33, 0.17, 0.28, 0.56])
            ax.imshow(image)
            ax.set_axis_off()
            _panel_title(ax, title, subtitle)
            if index < 2:
                self.fig.text(0.325 + index * 0.33, 0.45, "→", color=ACCENT, fontsize=40, ha="center", va="center")
        self.fig.text(0.5, 0.08, "The success check comes from the simulator run.  Cosmos shows the same run in new worlds;"
                      " GR00T was not re-tested inside them.", color=MUTED, fontsize=12.5, ha="center", va="center")

    def _worlds(self, state: MissionState) -> None:
        self._header("same brain-chosen GR00T episode  ·  re-rendered by NVIDIA Cosmos 3 (edge-controlled transfer)")
        self.fig.text(0.5, 0.905, "One brain decision, four worlds", color=TEXT, fontsize=24, fontweight="bold",
                      ha="center", va="center")
        self.fig.text(0.5, 0.868, "“push the plate to the front of the stove”  ·  the robot’s motion is the real "
                      "GR00T run; Cosmos repaints the world around it", color=MUTED, fontsize=13, ha="center")
        boxes = [[0.255, 0.44, 0.24, 0.40], [0.505, 0.44, 0.24, 0.40], [0.255, 0.02, 0.24, 0.40],
                 [0.505, 0.02, 0.24, 0.40]]
        for box, (key, image) in zip(boxes, state.world_frames.items()):
            ax = self.fig.add_axes(box)
            ax.imshow(image)
            ax.set_axis_off()
            ax.text(0.02, 0.97, state.world_titles.get(key, key), color=TEXT, fontsize=12, fontweight="bold",
                    transform=ax.transAxes, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor="none", alpha=0.75))
        if state.sim_frame is not None:
            ax = self.fig.add_axes([0.035, 0.30, 0.2, 0.36])
            ax.imshow(state.sim_frame)
            ax.set_axis_off()
            _panel_title(ax, "SOURCE", "the actual GR00T run (LIBERO sim)")
        for row, line in enumerate(state.history):
            self.fig.text(0.77, 0.62 - row * 0.05, line, color=TEXT if row == 0 else MUTED, fontsize=13)

    # ----------------------------------------------------------------- panels
    def _eeg(self, ax, state: MissionState) -> None:
        track = state.look
        _style(ax)
        title = f"looking at {state.intended}  (dataset label)  ·  seconds into the look" if track else "—"
        _panel_title(ax, "BRAIN · occipital EEG, 8 channels", title)
        ax.set_xlim(0, 2.0)
        ax.set_yticks([])
        if track is None:
            return
        if not state.live and track.decision_seconds < 2.0 - 1e-9:
            ax.axvline(track.decision_seconds, color=ACCENT, linestyle="--", linewidth=1.5)
        samples = int(round(state.look_elapsed * track.sampling_rate_hz))
        time = np.arange(samples) / track.sampling_rate_hz
        scale = 4 * np.median(np.abs(track.eeg)) + 1e-12
        for row, name in enumerate(self.channel_names):
            offset = len(self.channel_names) - row
            ax.plot(time, track.eeg[row, :samples] / scale + offset, color=ACCENT if state.live else MUTED,
                    linewidth=0.8, alpha=0.95 if state.live else 0.5)
            ax.text(-0.03, offset, name, color=MUTED, fontsize=8, ha="right", va="center")
        ax.set_ylim(0, len(self.channel_names) + 1)

    def _race(self, ax, state: MissionState) -> None:
        track = state.look
        _style(ax)
        if track is None:
            subtitle = "waiting for a look"
        elif state.live:
            subtitle = f"window {state.look_elapsed:.2f} s  ·  stops as soon as the lead is large enough"
        else:
            subtitle = f"decided at {track.decision_seconds:.1f} s  ·  margin {track.final['margin']:.2f}"
        _panel_title(ax, "DECODER · CCA score per target", subtitle)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, len(self.names) - 0.4)
        ax.set_yticks(range(len(self.names)), [f"{n}  {self.targets[n]:.1f} Hz" for n in self.names[::-1]],
                      color=TEXT, fontsize=10)
        ax.tick_params(axis="x", colors=MUTED, labelsize=8)
        elapsed = state.look_elapsed if state.live else (track.decision_seconds if track else 0.0)
        scores = track.scores_at(elapsed) if track is not None else np.full(len(self.names), np.nan)
        leader = int(np.nanargmax(scores)) if np.isfinite(scores).any() else None
        for row, name in enumerate(self.names):
            y = len(self.names) - 1 - row
            value = 0 if not np.isfinite(scores[row]) else scores[row]
            ax.barh(y, 1, color=PANEL, height=0.62)
            ax.barh(y, value, color=COLORS[name], alpha=0.95 if row == leader else 0.4, height=0.62)
            if np.isfinite(scores[row]):
                ax.text(min(value + 0.015, 0.93), y, f"{value:.2f}", color=TEXT, fontsize=9, va="center")

    def _history(self, ax, state: MissionState) -> None:
        _box(ax, "LOOKS  (every decision from a real recorded trial)")
        for row, line in enumerate(state.history[-6:]):
            ax.text(0.03, 0.74 - row * 0.12, line, color=TEXT if row == len(state.history[-6:]) - 1 else MUTED,
                    fontsize=10, family="monospace", transform=ax.transAxes)

    def _robot(self, ax, state: MissionState) -> None:
        if state.robot_frame is not None:
            ax.imshow(state.robot_frame)
        else:
            ax.add_patch(Rectangle((0, 0), 1, 1, color=PANEL, transform=ax.transAxes))
        ax.set_axis_off()
        _panel_title(ax, state.robot_title, state.robot_caption)
        if state.wipe is not None and 0 < state.wipe < 1:
            ax.axvline(state.wipe * (state.robot_frame.shape[1] - 1), color="white", linewidth=3)
            ax.text(0.02, 0.03, "NVIDIA COSMOS 3", color=ACCENT, fontsize=12, fontweight="bold", transform=ax.transAxes)
            ax.text(0.98, 0.03, "SIMULATOR", color=TEXT, fontsize=12, fontweight="bold", ha="right",
                    transform=ax.transAxes)
        if state.world_label:
            ax.text(0.02, 0.97, state.world_label, color=TEXT, fontsize=13, fontweight="bold", va="top",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor=BG, edgecolor=ACCENT, alpha=0.85))

    def _sim(self, ax, state: MissionState) -> None:
        if state.sim_frame is not None:
            ax.imshow(state.sim_frame)
        else:
            ax.add_patch(Rectangle((0, 0), 1, 1, color=PANEL, transform=ax.transAxes))
        ax.set_axis_off()
        _panel_title(ax, "THE ACTUAL RUN", "GR00T in the LIBERO simulator")

    def _pipeline(self, ax, state: MissionState) -> None:
        _box(ax, f"COMMAND {state.command} / {state.commands_total}")
        for row, name in enumerate(STAGES):
            y = 0.72 - row * 0.15
            done, active = row < state.stage, row == state.stage
            color = ACCENT if done else (TEXT if active else "#30363d")
            ax.text(0.05, y, "●" if done or active else "○", color=color, fontsize=13, transform=ax.transAxes,
                    va="center")
            ax.text(0.13, y, name, color=color, fontsize=11.5, fontweight="bold" if active else "normal",
                    transform=ax.transAxes, va="center")
        if state.instruction:
            ax.text(0.05, 0.02, f"“{state.instruction}”", color=ACCENT, fontsize=10, transform=ax.transAxes,
                    va="bottom", style="italic")

    def _stats(self, ax, state: MissionState) -> None:
        _box(ax, "MEASURED")
        for row, (key, value) in enumerate(state.stats.items()):
            ax.text(0.04, 0.68 - row * 0.2, key, color=MUTED, fontsize=9.5, transform=ax.transAxes)
            ax.text(0.96, 0.68 - row * 0.2, value, color=TEXT, fontsize=9.5, ha="right", fontweight="bold",
                    transform=ax.transAxes)

    def _card(self, state: MissionState) -> None:
        title, subtitle, lines = state.card
        overlay = self.fig.add_axes([0, 0, 1, 1])
        overlay.set_axis_off()
        overlay.add_patch(Rectangle((0, 0), 1, 1, color=BG, alpha=0.93 * state.fade, transform=overlay.transAxes))
        overlay.text(0.5, 0.72, title, color=ACCENT, fontsize=50, fontweight="bold", ha="center", va="center",
                     alpha=state.fade)
        overlay.text(0.5, 0.62, subtitle, color=TEXT, fontsize=22, ha="center", va="center", alpha=state.fade)
        for row, line in enumerate(lines):
            overlay.text(0.5, 0.5 - row * 0.055, line, color=MUTED if line.startswith("·") else TEXT, fontsize=16,
                         ha="center", va="center", alpha=state.fade)

    def close(self) -> None:
        plt.close(self.fig)


def _box(ax, title: str) -> None:
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02", facecolor=PANEL,
                                edgecolor="#30363d", transform=ax.transAxes))
    ax.text(0.04, 0.88, title, color=TEXT, fontsize=11.5, fontweight="bold", transform=ax.transAxes)
