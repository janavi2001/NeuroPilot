#!/usr/bin/env python3
"""Step 4e demo: recorded macaque M1/PMd spikes -> reach intention -> Franka pick.

The same robot pipeline as the SSVEP demo, with an intracortical decoder plugged
in. Story (each decision from a held-out val trial chosen by ground truth):
  1. A reach towards FIRST is decoded at movement onset; the robot starts reaching.
  2. During that reach, a lower-right reach (CANCEL) is decoded; the robot aborts.
  3. A reach towards SECOND is decoded; the robot grasps and lifts it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bci.intracortical_decoder import DecoderWindow, ReachIntentDecoder
from bci.nlb_data import DEFAULT_NWB, SCHEME_TARGETS, load_mc_maze
from interface.visualization.hero_renderer import ACCENT, COLORS
from interface.visualization.intracortical_renderer import (SECTORS, SIX_SECTORS, IntracorticalRenderer,
                                                            IntracorticalState, ReachPlayback)
from robot.franka_controller import FrankaPickController
from simulation.franka_environment import FrankaSimulation

FPS = 25
SLOWDOWN = 2.0  # playback runs at half speed so the spikes are readable
def preferred_order(trials, train: np.ndarray, window: DecoderWindow, classes) -> np.ndarray:
    """Unit display order by preferred reach direction (train trials only)."""
    counts = np.stack([trials.spike_counts(i, window.start, window.stop, window.stop - window.start)[:, 0]
                       for i in train])
    labels = trials.labels[train]
    xy = trials.target_xy[train]
    angles = np.array([np.angle(np.exp(1j * np.arctan2(xy[labels == c, 1], xy[labels == c, 0])).mean()) for c in classes])
    means = np.stack([counts[labels == c].mean(axis=0) for c in classes], axis=1)
    vector = means @ np.exp(1j * angles)
    return np.argsort(np.angle(vector))


class Demo:
    def __init__(self, args) -> None:
        self.args = args
        self.classes = SCHEME_TARGETS[args.scheme]
        self.trials = load_mc_maze(args.nwb, args.scheme)
        trials = self.trials
        self.train = np.flatnonzero(trials.split == "train")
        self.unused = {c: [int(i) for i in np.flatnonzero((trials.split == "val") & (trials.labels == c))]
                       for c in self.classes}
        evaluation = json.loads(args.eval.read_text()) if args.eval.is_file() else None
        w = evaluation["window"] if evaluation else {"start": -0.4, "stop": 0.0, "bin_seconds": 0.1}
        self.window = DecoderWindow(w["start"], w["stop"], w["bin_seconds"])
        self.decoder = ReachIntentDecoder(self.classes, self.window).fit(trials, self.train)
        self.course = [(end, ReachIntentDecoder(self.classes, DecoderWindow(end - 0.2, end, 0.1)).fit(trials, self.train))
                       for end in np.round(np.arange(-0.6, 0.001, 0.05), 2)]
        self.order = preferred_order(trials, self.train, self.window, self.classes)

        self.sim = FrankaSimulation(scene="six" if args.scheme == "six" else "three")
        self.controller = FrankaPickController(self.sim, on_step=self._tick)
        unique_targets = np.unique(trials.target_xy, axis=0)
        self.renderer = IntracorticalRenderer(self.sim, trials, unique_targets, SLOWDOWN, classes=self.classes,
                                              sectors=SIX_SECTORS if args.scheme == "six" else SECTORS)
        self.state = IntracorticalState(stats=load_stats(evaluation))
        self.steps_per_frame = round(1 / (FPS * self.sim.model.opt.timestep))
        self.step_count = 0
        self.playback_start_step = 0
        self.decided = False
        self.cancel_requested = False
        self.writer = None
        self.keyframes: list[tuple[str, np.ndarray]] = []
        self.log: list[dict] = []

    def _tick(self, phase: str) -> None:
        self.state.robot_phase = phase
        self.step_count += 1
        dt = self.sim.model.opt.timestep
        self.state.time = self.step_count * dt
        play = self.state.playback
        if play is not None:
            self.state.playback_t = play.start + (self.step_count - self.playback_start_step) * dt / SLOWDOWN
            if not self.decided and self.state.playback_t >= 0.0:
                self._decide(play)
            if self.state.playback_t >= play.stop:
                self.state.last, self.state.playback = play, None
        if self.step_count % self.steps_per_frame == 0:
            self.writer.write(self.renderer.frame(self.state))

    def idle(self, seconds: float) -> None:
        for _ in range(int(round(seconds / self.sim.model.opt.timestep))):
            self.sim.step()
            self._tick("idle" if self.state.robot_phase in ("idle", "home") else self.state.robot_phase)

    def start_trial(self, intended: str) -> bool:
        if not self.unused[intended]:
            self.state.history.append(f"no held-out {intended} trials left")
            return False
        index = self.unused[intended].pop(0)
        trials = self.trials
        onset = trials.move_onset[index]
        spikes = [trials.spike_times[u][(trials.spike_times[u] >= onset - 0.8) & (trials.spike_times[u] <= onset + 0.4)]
                  - onset for u in self.order]
        probabilities = [(float(end), decoder.predict_trial(trials, index)["scores"]) for end, decoder in self.course]
        final = {**self.decoder.predict_trial(trials, index), "window": (self.window.start, self.window.stop)}
        self.state.playback = ReachPlayback(index, intended, trials.target_xy[index], spikes,
                                            float(trials.go_cue[index] - onset), float(trials.target_on[index] - onset),
                                            probabilities, final)
        self.state.playback_t = -0.8
        self.playback_start_step = self.step_count
        self.decided = False
        self.state.banner, self.state.banner_color = f"monkey prepares a reach  ({intended} sector)", COLORS[intended]
        return True

    def _decide(self, play: ReachPlayback) -> None:
        self.decided = True
        decoded = play.final["intent"]
        moving = self.state.robot_phase in ("approach", "descend")
        if decoded == "CANCEL":
            outcome = "CANCEL · robot stops" if moving else "CANCEL · nothing to cancel"
            self.cancel_requested = moving
            self.state.selected, self.state.ghost_qpos = None, None
        else:
            outcome = f"SELECT {decoded}" + (" (ignored: robot busy)" if moving else "")
            if not moving:
                self.state.selected = decoded
                self.state.ghost_qpos = self.controller.plan_grasp(decoded)
        self.state.history.append(f"trial {play.trial:<3} reach {play.truth:<6} → {decoded:<6} p={play.final['score']:.2f}  {outcome.split(' ·')[0]}")
        self.state.banner = f"decoded {decoded} before the hand moved  →  {outcome}"
        self.state.banner_color = COLORS[decoded]
        self.log.append({"timestamp": datetime.now(timezone.utc).isoformat(), "dataset": "NLB MC_Maze_Large",
                         "trial": play.trial, "ground_truth": play.truth, "prediction": decoded,
                         "score": play.final["score"], "margin": play.final["margin"], "robot_phase": self.state.robot_phase,
                         "outcome": outcome})
        self.keyframes.append((f"decision_{len(self.log)}", self.renderer.frame(self.state)))

    def play(self, intended: str) -> str | None:
        if not self.start_trial(intended):
            return None
        while self.state.playback is not None:
            self.idle(1 / FPS)
        self.idle(0.6)
        return self.state.last.final["intent"]

    def run(self) -> dict:
        first, second = self.args.first.upper(), self.args.second.upper()
        self.state.card = ("NEUROPILOT", "Intention decoded from neurons, before the hand moves.",
                           "Recorded spiking of 162 neurons in macaque motor and premotor cortex (NLB MC_Maze, DANDI 000138),\n"
                           "replayed from held-out trials. The decoder sees only activity before movement onset.\n"
                           "Same robot pipeline as the EEG demo: only the decoder is swapped.")
        self.idle(3.0)
        self.keyframes.append(("title", self.renderer.frame(self.state)))
        self.state.card = None
        result = {"first": None, "aborted": False, "final_target": None, "pick": None}

        decoded = self.play(first)
        result["first"] = decoded
        if decoded not in self.sim.OBJECTS:
            return self.finish(result, second)
        self.cancel_requested = False
        self.start_trial("CANCEL")
        pick = self.controller.execute("pick", decoded, should_abort=lambda: self.cancel_requested)
        if not pick["aborted"]:
            while self.state.playback is not None:
                self.idle(1 / FPS)
            result.update(final_target=decoded, pick=pick)
            return self.finish(result, second)
        result["aborted"] = True
        self.keyframes.append(("cancel_mid_reach", self.renderer.frame(self.state)))
        self.controller.go_home(2.0)
        while self.state.playback is not None:
            self.idle(1 / FPS)
        self.state.robot_phase = "idle"
        decoded = self.play(second)
        if decoded not in self.sim.OBJECTS:
            return self.finish(result, second)
        self.state.banner, self.state.banner_color = f"robot picking {decoded}", ACCENT
        result.update(final_target=decoded, pick=self.controller.execute("pick", decoded))
        return self.finish(result, second)

    def finish(self, result: dict, intended: str) -> dict:
        pick = result["pick"]
        lifted = bool(pick and pick.get("lifted"))
        correct = result["final_target"] == intended
        result.update(intended=intended, lifted=lifted, target_correct=correct, success=lifted and correct)
        self.state.ghost_qpos = None
        if result["success"]:
            line = f"{result['final_target']} lifted {pick['lift_delta'] * 100:.0f} cm by a physical grasp"
            self.state.banner, self.state.banner_color = f"✓ {line}", ACCENT
            title = "PICKED BY INTENTION"
            self.keyframes.append(("lift", self.renderer.frame(self.state)))
        else:
            line = ("no object selected" if result["final_target"] is None else
                    f"robot lifted {result['final_target']}, intended {intended}" if lifted else "lift failed")
            self.state.banner, self.state.banner_color = f"✗ {line}", "#ff7b72"
            title = "NOT SUCCESSFUL"
            self.state.card_color = "#ff7b72"
        self.idle(2.0)
        stats = self.state.stats
        self.state.card = (title, line, f"held-out accuracy {stats.get('accuracy before movement', 'n/a')} on 125 trials · "
                           f"shuffled-label control {stats.get('shuffled-label control', 'n/a')}\n"
                           "Recorded monkey data replayed, not a human or a live implant. Linear decoder fitted on the NLB train split.")
        self.idle(4.0)
        self.keyframes.append(("end", self.renderer.frame(self.state)))
        return result


def load_stats(evaluation: dict | None) -> dict[str, str]:
    if evaluation is None:
        return {"evaluation": "run scripts/step4e_intracortical_eval.py"}
    val = evaluation["val"]
    course = {round(c["window_end"], 2): c["accuracy"] for c in evaluation["time_course"]}
    w = evaluation["window"]
    return {
        "accuracy before movement": f"{val['accuracy']:.1%}",
        "balanced across directions": f"{val['balanced_accuracy']:.1%}",
        "decoder window": f"{w['start'] * 1000:+.0f} to {w['stop'] * 1000:+.0f} ms of movement onset",
        "accuracy 600 ms before movement": f"{course.get(-0.6, float('nan')):.0%}",
        "shuffled-label control": f"{val['shuffled_label_control']:.1%}",
        "majority-class chance": f"{val['majority_chance']:.1%}",
    }


def main() -> int:
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--nwb", type=Path, default=DEFAULT_NWB)
    parser.add_argument("--scheme", choices=sorted(SCHEME_TARGETS), default="six",
                        help="'four' reproduces the 4-direction Step 4e demo")
    parser.add_argument("--eval", type=Path, default=None)
    parser.add_argument("--first", default="CUP")
    parser.add_argument("--second", default=None, help="default BOTTLE with six directions, APPLE with four")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=Path("artifacts/logs/step4e_intracortical_demo.jsonl"))
    args = parser.parse_args()
    six = args.scheme == "six"
    args.eval = args.eval or Path("artifacts/results/step4f_intracortical_eval_six.json" if six
                                  else "artifacts/results/step4e_intracortical_eval.json")
    args.second = args.second or ("BOTTLE" if six else "APPLE")
    args.video = args.video or Path(f"artifacts/videos/step4e/intracortical_demo{'_six' if six else ''}.mp4")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required")
    demo = Demo(args)
    args.video.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{demo.renderer.width}x{demo.renderer.height}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(args.video)], stdin=subprocess.PIPE)

    class Writer:
        def write(self, frame):
            process.stdin.write(frame.tobytes())

    demo.writer = Writer()
    try:
        result = demo.run()
    finally:
        process.stdin.close()
        process.wait()
        demo.renderer.close()
    shots = Path(f"artifacts/screenshots/step4e{'_six' if args.scheme == 'six' else ''}")
    shots.mkdir(parents=True, exist_ok=True)
    for label, frame in demo.keyframes:
        Image.fromarray(frame).save(shots / f"{label}.png")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a") as log:
        for row in demo.log:
            log.write(json.dumps(row) + "\n")
        log.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "robot": "franka_panda",
                              **{k: v for k, v in result.items() if k != "pick"},
                              "lift_delta": (result.get("pick") or {}).get("lift_delta"), "video": str(args.video)}) + "\n")
    for row in demo.log:
        print(f"  val trial {row['trial']}: reach {row['ground_truth']:<6} -> {row['prediction']:<6} "
              f"p={row['score']:.2f} (robot {row['robot_phase']}) -> {row['outcome']}")
    print(f"First: {result['first']}; aborted mid-reach: {result['aborted']}; final: {result['final_target']} "
          f"(intended {result['intended']}); lifted: {result['lifted']}")
    print(f"Video: {args.video}  ({demo.step_count * demo.sim.model.opt.timestep:.1f} s)")
    print(f"INTRACORTICAL DEMO: {'PASS' if result['success'] else 'FAIL'}")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
