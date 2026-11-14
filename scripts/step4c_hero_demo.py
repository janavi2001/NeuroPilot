#!/usr/bin/env python3
"""Step 4c hero demo: real SSVEP EEG -> confirmed selection -> Franka pick, with change of mind.

Story, every decision decoded from real recorded BETA trials chosen by ground truth:
  1. The subject looks at FIRST twice: preview (ghost arm), then confirm.
  2. The robot starts reaching; mid-reach the subject looks at CANCEL.
  3. If CANCEL is decoded, the robot aborts and returns home.
  4. The subject looks at SECOND twice; the robot grasps and lifts it.
If the decoder gets a look wrong, the video shows what really happens.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from bci.ssvep_data import load_beta_trials
from bci.ssvep_decoder import UNSURE, SSVEPDecoder
from bci.ssvep_stopping import EarlyStoppingDecoder, StoppingRule
from interface.ssvep_confirm import ConfirmingSelector
from interface.ssvep_selector import DEFAULT_CONFIG, load_ssvep_config
from interface.visualization.hero_renderer import ACCENT, COLORS, DemoState, HeroRenderer
from interface.visualization.look_track import LookTrack, tracks_for_sequence
from robot.franka_controller import FrankaPickController
from simulation.franka_environment import FrankaSimulation

FPS = 25
DECISION_HOLD = 0.9  # seconds a decision stays on screen before the next look


def eeg_channels_with_positions(subject: int):
    """BETA EEG channel names that have scalp coordinates, plus their MNE info."""
    import mne
    from moabb.datasets import Liu2020BETA

    raw = Liu2020BETA().get_data(subjects=[subject])[subject]["0"]["0"]
    raw.pick("eeg")
    montage = raw.get_montage()
    positions = montage.get_positions()["ch_pos"] if montage is not None else {}
    names = [name for name in raw.ch_names if name in positions and np.isfinite(positions[name]).all()]
    raw.pick(names)
    return names, raw.info


class Demo:
    def __init__(self, args) -> None:
        self.args = args
        self.config = load_ssvep_config(args.config)
        config = self.config
        self.eeg, labels, metadata = load_beta_trials(args.subject, config.channels, config.window_seconds,
                                                      list(config.targets.values()))
        all_names, self.eeg_info = eeg_channels_with_positions(args.subject)
        self.eeg_all, labels_all, _ = load_beta_trials(args.subject, all_names, config.window_seconds,
                                                       list(config.targets.values()))
        if not np.array_equal(labels, labels_all):
            raise RuntimeError("Decoder-channel and all-channel trial orders differ")
        frequency_to_name = {frequency: name for name, frequency in config.targets.items()}
        self.truths = [frequency_to_name[float(label)] for label in labels]
        self.unused = tracks_for_sequence(self.truths)
        self.decoder = SSVEPDecoder(config.targets, len(config.channels), metadata["sampling_rate_hz"],
                                    config.window_seconds, config.harmonics, config.margin_threshold)
        self.selector = ConfirmingSelector(config.objects, config.cancel_target)
        self.rule = None if args.fixed_window else StoppingRule.from_yaml(args.config)
        self.stopper = None if self.rule is None else EarlyStoppingDecoder(
            config.targets, len(config.channels), self.rule, metadata["sampling_rate_hz"],
            config.window_seconds, config.harmonics, config.margin_threshold)

        self.sim = FrankaSimulation()
        self.controller = FrankaPickController(self.sim, on_step=self._tick)
        self.state = DemoState(subject=args.subject, stats=load_stats(early=not args.fixed_window))
        self.renderer = HeroRenderer(self.sim, config.targets, list(config.channels), self.eeg_info,
                                     config.margin_threshold, stopping_rule=self.rule)
        self.steps_per_frame = round(1 / (FPS * self.sim.model.opt.timestep))
        self.step_count = 0
        self.look_start_step = 0
        self.pending_decision: tuple[LookTrack, str] | None = None
        self.cancel_requested = False
        self.writer = None
        self.keyframes: list[tuple[str, np.ndarray]] = []
        self.log: list[dict] = []

    # ------------------------------------------------------------ timing
    def _tick(self, phase: str) -> None:
        self.state.robot_phase = phase
        self.step_count += 1
        self.state.time = self.step_count * self.sim.model.opt.timestep
        if self.state.look is not None:
            self.state.look_elapsed = (self.step_count - self.look_start_step) * self.sim.model.opt.timestep
            if self.state.look_elapsed >= self.state.look.decision_seconds - 1e-9:
                self._finish_look()
        if self.step_count % self.steps_per_frame == 0:
            self.writer.write(self.renderer.frame(self.state))

    def idle(self, seconds: float, phase: str = "idle") -> None:
        for _ in range(int(round(seconds / self.sim.model.opt.timestep))):
            self.sim.step()
            self._tick(phase)

    def keyframe(self, label: str) -> None:
        self.keyframes.append((label, self.renderer.frame(self.state)))

    # ------------------------------------------------------------- looks
    def start_look(self, intended: str) -> bool:
        if not self.unused.get(intended):
            self.state.history.append(f"no recorded {intended} trials left")
            return False
        index = self.unused[intended].pop(0)
        self.state.look = LookTrack.build(intended, index + 1, self.eeg[index], self.eeg_all[index], self.decoder, FPS,
                                          stopper=self.stopper)
        self.state.look_elapsed = 0.0
        self.look_start_step = self.step_count
        self.state.banner = f"subject looks at {intended}"
        self.state.banner_color = COLORS[intended]
        return True

    def _finish_look(self) -> None:
        track = self.state.look
        prediction = track.final
        before = self.selector.state
        after = self.selector.apply(prediction)
        decided = prediction["intent"]
        margin = prediction["margin"]
        line = f"#{len(self.log) + 1:<2} saw {track.intended:<6} → {decided:<6} @{track.decision_seconds:.1f}s"
        if after.event == "confirm":
            outcome, color = f"CONFIRMED {after.confirmed}", ACCENT
        elif after.event in ("candidate", "switch"):
            outcome, color = f"preview {after.candidate} · look again to confirm", COLORS[after.candidate]
        elif after.event == "cancel":
            outcome, color = "CANCEL", COLORS["CANCEL"]
            self.cancel_requested = True
        else:
            outcome, color = "UNSURE · no change", "#8b949e"
        self.state.history.append(f"{line}  {outcome.split(' ·')[0]}")
        self.state.banner, self.state.banner_color = f"decoded {decided}  →  {outcome}", color
        self.state.candidate, self.state.confirmed = after.candidate, after.confirmed
        self.state.ghost_qpos = self.controller.plan_grasp(after.candidate) if after.candidate else (
            None if after.event == "cancel" else self.state.ghost_qpos)
        self.log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(), "subject": self.args.subject,
            "trial": track.trial_number, "ground_truth": track.intended, "prediction": decided,
            "score": prediction["score"], "margin": margin, "decision_seconds": track.decision_seconds,
            "early_stop": track.decision_seconds < self.config.window_seconds - 1e-9, "event": after.event,
            "candidate": after.candidate, "confirmed": after.confirmed, "previous_candidate": before.candidate,
            "robot_phase": self.state.robot_phase,
        })
        self.state.last_look, self.state.look = track, None

    def look(self, intended: str) -> bool:
        if not self.start_look(intended):
            return False
        while self.state.look is not None:
            self.idle(1 / FPS)
        self.keyframe(f"decision_{len(self.log)}")
        self.idle(DECISION_HOLD)
        return True

    def acquire(self, intended: str, max_looks: int = 4) -> str | None:
        """Look at ``intended`` until two agreeing looks confirm an object or trials run out."""
        for _ in range(max_looks):
            if not self.look(intended):
                return None
            if self.selector.state.confirmed is not None:
                return self.selector.state.confirmed
        return None

    # ------------------------------------------------------------- story
    def run(self) -> dict:
        args = self.args
        first, second = args.first.upper(), args.second.upper()
        self.state.card = ("NEUROPILOT", "Look at an object. A robot picks it up.",
                           f"Real EEG recorded from BETA SSVEP subject S{args.subject}, replayed through a training-free CCA decoder.\n"
                           "Each object flickers at its own frequency; the visual cortex follows it.\n"
                           "Flicker is shown 8× slower here for photosensitivity safety.")
        self.idle(4.0)
        self.keyframe("title")
        self.state.card = None

        result = {"first_confirmed": None, "aborted": False, "final_target": None, "pick": None}
        target = self.acquire(first)
        result["first_confirmed"] = target
        if target is None:
            return self.finish(result, intended=second)
        self.state.banner, self.state.banner_color = f"robot reaching for {target}", ACCENT
        self.selector.reset()
        self.state.confirmed = target
        self.cancel_requested = False
        if not args.no_change_of_mind:
            self.start_look("CANCEL")
        pick = self.controller.execute("pick", target, should_abort=lambda: self.cancel_requested)
        if pick["aborted"]:
            result["aborted"] = True
            self.keyframe("cancel_mid_reach")
            self.state.banner, self.state.banner_color = f"CANCEL decoded mid-reach · robot stops and returns", COLORS["CANCEL"]
            self.state.confirmed, self.state.ghost_qpos = None, None
            self.controller.go_home(2.0)
            self.state.robot_phase = "idle"
            self.selector.reset()
            target = self.acquire(second)
            if target is None:
                return self.finish(result, intended=second)
            self.state.banner, self.state.banner_color = f"robot picking {target}", ACCENT
            pick = self.controller.execute("pick", target)
            intended = second
        else:
            while self.state.look is not None:  # the CANCEL look may still be finishing
                self.idle(1 / FPS)
            intended = first if args.no_change_of_mind else second
        result["final_target"], result["pick"] = target, pick
        self.state.ghost_qpos = None
        self.keyframe("lift")
        return self.finish(result, intended=intended)

    def finish(self, result: dict, intended: str) -> dict:
        pick = result["pick"]
        lifted = bool(pick and pick["lifted"])
        correct = result["final_target"] == intended
        result.update(intended=intended, lifted=lifted, target_correct=correct, success=lifted and correct)
        if result["success"]:
            line = f"{result['final_target']} lifted {pick['lift_delta'] * 100:.0f} cm by a physical grasp"
            self.state.banner, self.state.banner_color = f"✓ {line}", ACCENT
            title = "PICKED BY LOOKING"
        else:
            reason = "no confirmed selection" if result["final_target"] is None else (
                f"robot lifted {result['final_target']}, subject meant {intended}" if lifted else "lift failed")
            self.state.banner, self.state.banner_color = f"✗ {reason}", "#ff7b72"
            title, line = "NOT SUCCESSFUL", reason
            self.state.card_color = "#ff7b72"
        self.idle(2.5, phase="hold" if lifted else "idle")
        stats = self.state.stats
        self.state.card = (title, line, f"4 targets · held-out accuracy {stats.get('single-look accuracy', 'n/a')} · "
                           f"wrong actions {stats.get('wrong actions, two-look confirm', 'n/a')} with two-look confirm\n"
                           "Replayed EEG from one subject; not a live headset. Motion is scripted IK, not a learned policy.")
        self.idle(4.0, phase="hold" if lifted else "idle")
        self.keyframe("end")
        return result


def load_stats(early: bool = False) -> dict[str, str]:
    stats = {}
    stopping = Path("artifacts/results/step4d_early_stop_eval.json")
    single = Path("artifacts/results/step4b_ssvep_eval.json")
    confirm = Path("artifacts/results/step4c_confirm_eval.json")
    if single.is_file():
        data = json.loads(single.read_text())
        stats["single-look accuracy"] = f"{data['test_without_threshold']['raw_accuracy']:.1%}"
        stats["wrong actions, single look"] = f"{data['test']['wrong_action_fraction']:.1%}"
        stats["decode time per look"] = f"{data['test']['mean_runtime_ms']:.2f} ms"
    if confirm.is_file():
        data = json.loads(confirm.read_text())
        stats["wrong actions, two-look confirm"] = f"{data['two_look_confirm']['wrong_action_rate']:.1%}"
        stats["information rate"] = f"{data['itr']['bits_per_minute']:.0f} bits/min"
    if early and stopping.is_file():
        data = json.loads(stopping.read_text())["test"]
        # Report the error rates of the decoder actually running, not the fixed-window ones.
        stats["single-look accuracy"] = f"{data['early']['forced_accuracy']:.1%}"
        stats["wrong actions, single look"] = f"{data['early']['wrong_action_rate']:.1%}"
        stats["wrong actions, two-look confirm"] = (f"{data['confirm_early']['wrong_action_rate']:.1%} "
                                                    f"({data['confirm_early']['wrong']}/{data['confirm_early']['pairs']})")
        stats["mean decision time, early stop"] = f"{data['early']['mean_decision_seconds']:.2f} s (vs 2.00 s)"
        stats["information rate"] = (f"{data['fixed']['bits_per_minute']:.0f} → "
                                     f"{data['early']['bits_per_minute']:.0f} bits/min with early stop")
    return stats


def main() -> int:
    import subprocess
    import shutil
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--first", default="CUP", help="object chosen first (then cancelled mid-reach)")
    parser.add_argument("--second", default="PHONE", help="object chosen after the change of mind")
    parser.add_argument("--no-change-of-mind", action="store_true", help="skip the mid-reach CANCEL look")
    parser.add_argument("--fixed-window", action="store_true", help="always decode the full 2 s window (no early stopping)")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=Path("artifacts/logs/step4c_hero_demo.jsonl"))
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required")

    demo = Demo(args)
    video = args.video or Path(f"artifacts/videos/step4c/hero_S{args.subject}.mp4")
    video.parent.mkdir(parents=True, exist_ok=True)
    width, height = demo.renderer.width, demo.renderer.height
    process = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(video)],
        stdin=subprocess.PIPE)

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

    shots = Path(f"artifacts/screenshots/step4c_hero/S{args.subject}")
    shots.mkdir(parents=True, exist_ok=True)
    for label, frame in demo.keyframes:
        Image.fromarray(frame).save(shots / f"{label}.png")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a") as log:
        for row in demo.log:
            log.write(json.dumps(row) + "\n")
        pick = result.get("pick") or {}
        log.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "subject": args.subject,
                              "robot": "franka_panda", **{k: v for k, v in result.items() if k != "pick"},
                              "lift_delta": pick.get("lift_delta"), "video": str(video)}) + "\n")

    for row in demo.log:
        print(f"  trial {row['trial']:2d}: looked at {row['ground_truth']:<6} -> {row['prediction']:<6} "
              f"at {row['decision_seconds']:.1f} s, margin {row['margin']:.3f} -> {row['event']}"
              + (f" (robot was {row['robot_phase']})" if row["robot_phase"] not in ("idle",) else ""))
    print(f"First confirmed: {result['first_confirmed']}; aborted mid-reach: {result['aborted']}")
    print(f"Final target: {result['final_target']} (intended {result['intended']}); lifted: {result['lifted']}")
    if result.get("pick") and not result["pick"].get("aborted"):
        print(f"Lift: {result['pick']['initial_z']:.3f} -> {result['pick']['final_z']:.3f} m; "
              f"both fingers in contact: {result['pick']['fingers_in_contact_after_lift']}")
    print(f"Video: {video}  ({demo.step_count * demo.sim.model.opt.timestep:.1f} s)")
    print(f"Keyframes: {shots}")
    print(f"HERO DEMO: {'PASS' if result['success'] else 'FAIL'}")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
