#!/usr/bin/env python3
"""Step 4c: real BETA SSVEP EEG -> 4-target selection -> Franka Panda physical pick."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco

from bci.ssvep_data import load_beta_trials
from bci.ssvep_decoder import SSVEPDecoder
from interface.ssvep_replay import intended_final, replay_sequence
from interface.ssvep_selector import DEFAULT_CONFIG, SSVEPSelector, load_ssvep_config
from robot.franka_controller import FrankaPickController
from simulation.franka_environment import FrankaSimulation


class VideoWriter:
    """Pipe RGB frames to ffmpeg as an H.264 MP4."""

    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required for --video; install it or pass --no-video")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)],
            stdin=subprocess.PIPE,
        )

    def write(self, frame) -> None:
        self.process.stdin.write(frame.tobytes())

    def close(self) -> None:
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("ffmpeg failed to write the video")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--subject", type=int, default=1, help="BETA subject to replay (1-70)")
    parser.add_argument("--sequence", nargs="+", default=["PHONE", "CANCEL", "CUP"],
                        help="Targets the subject was really looking at; each uses that target's next recorded trial")
    parser.add_argument("--video", type=Path, default=None, help="MP4 path (default artifacts/videos/step4c/S<subject>_<target>.mp4)")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--log", type=Path, default=Path("artifacts/logs/step4c_franka_demo.jsonl"))
    parser.add_argument("--gui", action="store_true", help="Open the MuJoCo viewer (run with .venv/bin/mjpython on macOS)")
    args = parser.parse_args()

    config = load_ssvep_config(args.config)
    sequence = [name.upper() for name in args.sequence]
    unknown = [name for name in sequence if name not in config.targets]
    if unknown:
        parser.error(f"Unknown targets {unknown}; choose from {list(config.targets)}")

    eeg, labels, metadata = load_beta_trials(args.subject, config.channels, config.window_seconds, list(config.targets.values()))
    frequency_to_name = {frequency: name for name, frequency in config.targets.items()}
    truths = [frequency_to_name[float(label)] for label in labels]
    decoder = SSVEPDecoder(config.targets, len(config.channels), metadata["sampling_rate_hz"],
                           config.window_seconds, config.harmonics, config.margin_threshold)
    selector = SSVEPSelector(config.objects, config.cancel_target)
    print(f"Dataset: {metadata['dataset']} subject S{args.subject}; window {config.window_seconds:.1f} s; "
          f"targets " + ", ".join(f"{n}={f:.1f} Hz" for n, f in config.targets.items()))

    events = replay_sequence(decoder, selector, eeg, truths, sequence)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log = args.log.open("a")
    for event in events:
        p = event.prediction
        print(f"  looked at {event.intended:<6} (trial {event.trial_number}) -> {p['intent']:<6} "
              f"margin {p['margin']:.3f} -> selected={event.selected_object}")
        log.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(), "decoder": "CCA", "dataset": "BETA",
            "subject": args.subject, "trial": event.trial_number, "ground_truth": event.intended,
            "prediction": p["intent"], "score": p["score"], "margin": p["margin"],
            "selected_object": event.selected_object, "latency_ms": event.latency_ms,
        }) + "\n")
    target = selector.state.selected_object
    intended = intended_final(sequence, config.cancel_target)
    print(f"Final selection: {target}  (intended: {intended})")
    if target is None:
        print("No object selected; no robot action issued.")
        log.close()
        return 0

    simulation = FrankaSimulation(gui=args.gui)
    writer = renderer = None
    video_path = args.video or Path(f"artifacts/videos/step4c/S{args.subject}_{target.lower()}.mp4")
    if not args.no_video:
        width, height = 1280, 720
        renderer = mujoco.Renderer(simulation.model, height, width)
        writer = VideoWriter(video_path, width, height, args.fps)
    steps_per_frame = max(1, round(1 / (args.fps * simulation.model.opt.timestep)))
    step_count = 0

    def on_step(_phase: str) -> None:
        nonlocal step_count
        step_count += 1
        if writer is not None and step_count % steps_per_frame == 0:
            writer.write(simulation.render(renderer, "overview"))
        if args.gui:
            time.sleep(simulation.model.opt.timestep)

    try:
        start = time.perf_counter()
        result = FrankaPickController(simulation, on_step).execute(action="pick", target=target)
        robot_seconds = time.perf_counter() - start
    finally:
        if writer is not None:
            writer.close()
            renderer.close()
    print(f"Robot: Franka Panda, physical grasp (no attachment)")
    print(f"Grasp contact: {'both fingers' if result['grasp_success'] else 'FAILED'}")
    print(f"Object height: {result['initial_z']:.3f} -> {result['final_z']:.3f} m (delta {result['lift_delta']:.3f})")
    print(f"Object in gripper after lift: {result['fingers_in_contact_after_lift']} "
          f"(object-gripper distance {result['object_tcp_distance']:.3f} m)")
    print(f"Other objects disturbed by at most {result['max_other_object_shift'] * 100:.1f} cm")
    print(f"Object lifted: {'YES' if result['lifted'] else 'NO'}")
    target_correct = target == intended
    log.write(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(), "robot": "franka_panda", "selected_object": target,
        "intended_object": intended, "robot_action": f"pick {target}", "lifted": result["lifted"],
        "target_correct": target_correct, "success": result["lifted"] and target_correct,
        "lift_delta": result["lift_delta"], "robot_wall_seconds": robot_seconds,
        "video": None if args.no_video else str(video_path),
    }) + "\n")
    log.close()
    if writer is not None:
        print(f"Video: {video_path}")
    if args.gui:
        input("Press Enter to close the MuJoCo window...")
    simulation.close()
    if not result["lifted"]:
        print("SSVEP -> FRANKA TEST: FAIL (physical lift verification failed)")
        return 1
    if not target_correct:
        print(f"SSVEP -> FRANKA TEST: FAIL (decoder selected {target}; subject looked at {intended}; robot lifted {target})")
        return 1
    print("SSVEP -> FRANKA TEST: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
