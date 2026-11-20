"""End-to-end demo frame: real EEG decision -> GR00T action -> Cosmos worlds, one 1920x1080 frame at a time."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from .hero_renderer import ACCENT, BG, MUTED, PANEL, TEXT, _panel_title, _style  # noqa: E402
from .look_track import LookTrack  # noqa: E402

COLORS = {"BOWL": "#f0883e", "PLATE": "#58a6ff", "CHEESE": "#e3d026", "WINE": "#ff6b6b", "STOVE": "#39c5cf",
          "CANCEL": "#d2a8ff", "UNSURE": MUTED}
STAGES = ("LOOK", "CONFIRM", "INSTRUCTION", "GR00T ACTS", "CHECKED")
WARN = "#f0883e"
UNSPOKEN = "#484f58"
DIM = 0.62  # opacity of the veil over panels outside the spotlight


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
    focus: tuple[str, ...] = ()  # spotlighted panel keys; everything else is dimmed
    caption: list[tuple[str, str | None, bool]] = field(default_factory=list)  # (word, keyword colour, spoken)
    stimulus: bool = False  # show "what the person sees" in the top-right slot
    clock: float = 0.0  # seconds since start, drives the stimulus flicker
    speed_note: str = ""  # e.g. "replayed at 0.15x speed", shown on the EEG panel


class MissionRenderer:
    def __init__(self, targets: dict[str, float], channel_names: list[str], threshold: float,
                 width: int = 1920, height: int = 1080) -> None:
        self.targets = targets
        self.names = list(targets)
        self.channel_names = channel_names
        self.threshold = threshold
        self.fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=BG)
        self._rects: dict[str, list[float]] = {}

    def frame(self, state: MissionState) -> np.ndarray:
        fig = self.fig
        fig.clf()
        fig.set_facecolor(BG)
        self._rects = {}
        if state.scene == "kitchen":
            self._kitchen(state)
        elif state.scene == "worlds":
            self._worlds(state)
        elif state.scene == "explain":
            self._explain(state)
        elif state.scene == "idea":
            self._idea(state)
        if state.card is not None:
            self._card(state)
        if state.focus:
            self._spotlight(state.focus)
        if state.caption:
            self._captions(state.caption)
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
        layout = {"eeg": [0.035, 0.655, 0.3, 0.235], "race": [0.075, 0.375, 0.26, 0.205],
                  "history": [0.015, 0.105, 0.32, 0.2], "robot": [0.355, 0.17, 0.395, 0.72],
                  "sim": [0.77, 0.6, 0.215, 0.29], "pipeline": [0.77, 0.31, 0.215, 0.25],
                  "stats": [0.77, 0.105, 0.215, 0.17]}
        self._eeg(self._axes("eeg", layout["eeg"], title=True), state)
        self._race(self._axes("race", layout["race"], title=True, left=0.06), state)
        self._history(self._axes("history", layout["history"]), state)
        self._robot(self._axes("robot", layout["robot"], title=True), state)
        if state.stimulus:
            self._stimulus(self._axes("stimulus", layout["sim"], title=True), state)
        elif state.sim_frame is not None:
            self._sim(self._axes("sim", layout["sim"], title=True), state)
        self._pipeline(self._axes("pipeline", layout["pipeline"]), state)
        self._stats(self._axes("stats", layout["stats"]), state)
        if state.banner:
            self.fig.text(0.5525, 0.128, state.banner, color=state.banner_color, fontsize=18, fontweight="bold",
                          ha="center", va="center",
                          bbox=dict(boxstyle="round,pad=0.45", facecolor=PANEL, edgecolor=state.banner_color,
                                    linewidth=2))
            self._rects["banner"] = [0.36, 0.1, 0.385, 0.056]

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
            ax = self._axes(f"x{index + 1}", [0.03 + index * 0.33, 0.19, 0.28, 0.54], title=True)
            ax.imshow(image)
            ax.set_axis_off()
            _panel_title(ax, title, subtitle)
            if index < 2:
                self.fig.text(0.325 + index * 0.33, 0.45, "→", color=ACCENT, fontsize=40, ha="center", va="center")
        self.fig.text(0.5, 0.135, "The success check comes from the simulator run.  Cosmos shows the same run in new worlds;"
                      " GR00T was not re-tested inside them.", color=MUTED, fontsize=12.5, ha="center", va="center")

    def _worlds(self, state: MissionState) -> None:
        self._header("same brain-chosen GR00T episode  ·  re-rendered by NVIDIA Cosmos 3 (edge-controlled transfer)")
        self.fig.text(0.5, 0.905, "One brain decision, four worlds", color=TEXT, fontsize=24, fontweight="bold",
                      ha="center", va="center")
        self.fig.text(0.5, 0.868, "“push the plate to the front of the stove”  ·  the robot’s motion is the real "
                      "GR00T run; Cosmos repaints the world around it", color=MUTED, fontsize=13, ha="center")
        boxes = [[0.27, 0.475, 0.22, 0.37], [0.51, 0.475, 0.22, 0.37], [0.27, 0.095, 0.22, 0.37],
                 [0.51, 0.095, 0.22, 0.37]]
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

    def _idea(self, state: MissionState) -> None:
        self._header("the idea")
        self.fig.text(0.5, 0.8, "The brain says WHAT.  The robot works out HOW.", color=TEXT, fontsize=30,
                      fontweight="bold", ha="center", va="center")
        boxes = [("BRAIN", "chooses a goal", "“the bowl”", "#d2a8ff"),
                 ("INSTRUCTION", "plain language", "“put the bowl on the stove”", ACCENT),
                 ("ROBOT", "plans every joint and grip", "NVIDIA GR00T", "#58a6ff")]
        for index, (title, subtitle, example, color) in enumerate(boxes):
            x = 0.06 + index * 0.31
            ax = self._axes(f"i{index + 1}", [x, 0.33, 0.26, 0.3])
            ax.set_axis_off()
            ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.04", facecolor=PANEL,
                                        edgecolor=color, linewidth=2.5, transform=ax.transAxes))
            ax.text(0.5, 0.74, title, color=color, fontsize=26, fontweight="bold", ha="center", transform=ax.transAxes)
            ax.text(0.5, 0.5, subtitle, color=TEXT, fontsize=16, ha="center", transform=ax.transAxes)
            ax.text(0.5, 0.24, example, color=MUTED, fontsize=15, ha="center", style="italic", transform=ax.transAxes)
            if index < 2:
                self.fig.text(x + 0.285, 0.48, "→", color=ACCENT, fontsize=44, ha="center", va="center")
        self.fig.text(0.5, 0.21, "No joint-by-joint control from the brain: a small, reliable signal is enough.",
                      color=MUTED, fontsize=15, ha="center", va="center")

    # ----------------------------------------------------------------- panels
    def _stimulus(self, ax, state: MissionState) -> None:
        """Six targets flickering at their SSVEP rates, slowed 8x (photosensitivity-safe)."""
        ax.set_axis_off()
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 2)
        _panel_title(ax, "WHAT THE PERSON SEES", "each target flickers at its own rate · shown 8× slower")
        for index, name in enumerate(self.names):
            col, row = index % 3, 1 - index // 3
            pulse = 0.5 + 0.5 * np.sin(2 * np.pi * self.targets[name] / 8.0 * state.clock)
            ax.add_patch(FancyBboxPatch((col + 0.08, row + 0.1), 0.84, 0.8, boxstyle="round,pad=0,rounding_size=0.08",
                                        facecolor=COLORS[name], alpha=0.15 + 0.6 * pulse, edgecolor=COLORS[name]))
            ax.text(col + 0.5, row + 0.58, name, color=TEXT, fontsize=12, fontweight="bold", ha="center", va="center")
            ax.text(col + 0.5, row + 0.3, f"{self.targets[name]:.1f} Hz", color=TEXT, fontsize=9, ha="center",
                    va="center")
            if name == state.intended:
                ax.add_patch(FancyBboxPatch((col + 0.02, row + 0.04), 0.96, 0.92, boxstyle="round,pad=0,rounding_size=0.1",
                                            fill=False, edgecolor="white", linewidth=3))
                ax.text(col + 0.5, row + 0.8, "LOOKING", color="white", fontsize=8.5, fontweight="bold",
                        ha="center", va="center")

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
        if state.speed_note:
            ax.text(0.99, 0.97, state.speed_note, color=WARN, fontsize=9, ha="right", va="top", transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=BG, edgecolor=WARN, alpha=0.85))
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
        big = len(lines) <= 2  # a short card is a statement; a long one is a fact sheet
        for row, line in enumerate(lines):
            overlay.text(0.5, 0.5 - row * (0.075 if big else 0.055), line,
                         color=MUTED if line.startswith("·") else TEXT, fontsize=26 if big else 16,
                         ha="center", va="center", alpha=state.fade)

    # ---------------------------------------------------------------- overlays
    def _axes(self, key: str, rect: list[float], title: bool = False, left: float = 0.0):
        """Add an axes and remember its footprint (with its title and tick labels) for the spotlight."""
        x, y, w, h = rect
        self._rects[key] = [x - left - 0.006, y - (0.03 if title else 0.006), w + left + 0.012,
                            h + (0.075 if title else 0.012)]
        return self.fig.add_axes(rect)

    def _spotlight(self, focus: tuple[str, ...]) -> None:
        """Glow around the panels being talked about; a veil over the others."""
        overlay = self.fig.add_axes([0, 0, 1, 1], zorder=50)
        overlay.set_axis_off()
        overlay.set_xlim(0, 1)
        overlay.set_ylim(0, 1)
        for key, (x, y, w, h) in self._rects.items():
            if key in focus:
                for width, alpha in ((14, 0.12), (7, 0.25), (2.5, 1.0)):
                    overlay.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01",
                                                     fill=False, edgecolor=ACCENT, linewidth=width, alpha=alpha))
            else:
                overlay.add_patch(Rectangle((x, y), w, h, facecolor=BG, alpha=DIM, edgecolor="none"))

    def _captions(self, words: list[tuple[str, str | None, bool]]) -> None:
        """Karaoke subtitles: spoken words bright (keywords in colour), upcoming words dim; at most two lines."""
        lines, current, length = [], [], 0
        for word in words:
            if current and length + len(word[0]) > 105:
                lines.append(current)
                current, length = [], 0
            current.append(word)
            length += len(word[0]) + 1
        if current:
            lines.append(current)
        last_spoken = max((i for i, line in enumerate(lines) if any(w[2] for w in line)), default=0)
        start = min(max(0, last_spoken - 1), max(0, len(lines) - 2))
        rows = []
        for line in lines[start:start + 2]:
            parts = [TextArea(word, textprops=dict(color=(color or TEXT) if spoken else UNSPOKEN, fontsize=19,
                                                  fontweight="bold" if spoken and color else "normal"))
                     for word, color, spoken in line]
            rows.append(HPacker(children=parts, align="baseline", pad=0, sep=7))
        box = AnchoredOffsetbox(loc="lower center", child=VPacker(children=rows, align="center", pad=0, sep=6),
                                frameon=True, bbox_to_anchor=(0.5, 0.008), bbox_transform=self.fig.transFigure,
                                borderpad=0, pad=0.55)
        box.patch.set(facecolor=BG, edgecolor="#30363d", alpha=0.92, boxstyle="round,pad=0.2,rounding_size=0.5")
        box.set_zorder(60)
        overlay = self.fig.add_axes([0, 0, 1, 1], zorder=60)
        overlay.set_axis_off()
        overlay.add_artist(box)

    def close(self) -> None:
        plt.close(self.fig)


def _box(ax, title: str) -> None:
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02", facecolor=PANEL,
                                edgecolor="#30363d", transform=ax.transAxes))
    ax.text(0.04, 0.88, title, color=TEXT, fontsize=11.5, fontweight="bold", transform=ax.transAxes)
