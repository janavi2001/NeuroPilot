#!/usr/bin/env python3
"""End-to-end demo video: real EEG decisions -> pretrained GR00T -> Cosmos worlds.

Nothing in the video is animated by hand:
  - every look replays the recorded BETA trial that Step 5a decoded, through
    the same growing-window CCA and early-stopping rule; the replayed decision
    must match the one logged in bci_jobs.json or the script stops;
  - the robot is the actual GR00T/LIBERO episode for that command (inset) and
    its Cosmos 3 re-render (main panel);
  - every number on screen is read from the stored evaluation files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bci.ssvep_data import load_beta_trials  # noqa: E402
from bci.ssvep_decoder import SSVEPDecoder  # noqa: E402
from bci.ssvep_stopping import EarlyStoppingDecoder, StoppingRule  # noqa: E402
from cosmos.controls import read_frames  # noqa: E402
from interface.ssvep_selector import load_ssvep_config  # noqa: E402
from interface.visualization.hero_renderer import ACCENT, MUTED, TEXT  # noqa: E402
from interface.visualization.look_track import LookTrack  # noqa: E402
from interface.visualization.mission_renderer import COLORS, MissionRenderer, MissionState  # noqa: E402

FPS = 25
SOURCE_FPS = 20  # GR00T/LIBERO and Cosmos clips
CONFIG = Path("interface/ssvep_targets_kitchen.yaml")
GR00T = Path("artifacts/step5_gr00t")
COSMOS = Path("artifacts/step5_cosmos")
RESULTS = Path("artifacts/results")
WORLD_ORDER = ["modern_kitchen", "warehouse", "office", "night_kitchen"]
EVENT_TEXT = {"candidate": ("PREVIEW · look again to confirm", None), "confirm": ("CONFIRMED", ACCENT),
              "unsure": ("UNSURE · lead too small, nothing happens", MUTED), "switch": ("new PREVIEW", None),
              "cancel": ("CANCEL · selection cleared", COLORS["CANCEL"])}


def cosmos_clip(label: str, world: str) -> Path:
    path = COSMOS / "remote_results" / f"{label}__{world}" / "vision.mp4"
    if not path.is_file():
        raise FileNotFoundError(f"missing Cosmos clip {path}")
    return path


def load_numbers() -> dict[str, str]:
    ssvep = json.loads((RESULTS / "step4f_ssvep_eval.json").read_text())["test"]
    early = json.loads((RESULTS / "step4f_early_stop_eval.json").read_text())["test"]["early"]
    confirm = json.loads((RESULTS / "step4f_confirm_eval.json").read_text())["two_look_confirm"]
    gr00t = json.loads((GR00T / "summary.json").read_text())
    cosmos = json.loads((COSMOS / "summary.json").read_text())
    baseline_ok = sum(v["successes"] for v in gr00t["baseline"].values())
    baseline_n = sum(v["episodes"] for v in gr00t["baseline"].values())
    return {
        "accuracy": f"{100 * ssvep['raw_accuracy']:.1f}%",
        "confirm_wrong": f"{100 * confirm['wrong_action_rate']:.1f}%",
        "confirm_detail": f"{confirm['wrong']}/{confirm['pairs']}",
        "bits": f"{early['bits_per_minute']:.0f} bits/min",
        "decision": f"{early['mean_decision_seconds']:.2f} s",
        "baseline": f"{baseline_ok} / {baseline_n}",
        "bci_correct": f"{sum(d['confirmed'] == d['intended'] for d in gr00t['bci'])} / {len(gr00t['bci'])}",
        "bci_success": f"{sum(d['robot_success'] for d in gr00t['bci'])} / {len(gr00t['bci'])}",
        "edge_f1": f"{cosmos['mean_edge_f1']:.3f}", "edge_f1_sim": f"{cosmos['mean_edge_f1_original_sim']:.3f}",
        "cosmos_clips": str(cosmos["clips_ok"]),
    }


def build_tracks(decisions: list[dict], subject: int):
    """LookTrack per logged look, replayed from the same recorded trial; must reproduce the logged decision."""
    config = load_ssvep_config(CONFIG)
    eeg, _, metadata = load_beta_trials(subject, config.channels, config.window_seconds, list(config.targets.values()))
    fs = metadata["sampling_rate_hz"]
    decoder = SSVEPDecoder(config.targets, len(config.channels), fs, config.window_seconds, config.harmonics,
                           config.margin_threshold)
    stopper = EarlyStoppingDecoder(config.targets, len(config.channels), StoppingRule.from_yaml(CONFIG), fs,
                                   config.window_seconds, config.harmonics, config.margin_threshold)
    tracks = []
    for decision in decisions:
        row = []
        for look in decision["looks"]:
            trial = eeg[look["trial"] - 1]
            track = LookTrack.build(look["ground_truth"], look["trial"], trial, trial, decoder, FPS, stopper=stopper)
            if (track.final["intent"], track.decision_seconds) != (look["intent"], look["decision_seconds"]):
                raise RuntimeError(f"replay of trial {look['trial']} gave {track.final['intent']}@"
                                   f"{track.decision_seconds}s, logged {look['intent']}@{look['decision_seconds']}s")
            row.append((look, track))
        tracks.append(row)
    return config, tracks


COMMAND_WORLDS = {"bci_02_PLATE": "warehouse", "bci_03_CHEESE": "office", "bci_04_WINE": "night_kitchen",
                  "bci_05_STOVE": "modern_kitchen"}  # command 1 (BOWL) is shown in the plain simulator, then explained
WORLD_TITLES = {"modern_kitchen": "modern kitchen · daylight", "warehouse": "warehouse · fluorescent",
                "office": "office break room", "night_kitchen": "kitchen at night"}
WIPE_SECONDS = 1.0


def wiped(cosmos: np.ndarray, sim: np.ndarray, fraction: float) -> np.ndarray:
    """Cosmos on the left of the wipe line, simulator on the right."""
    column = int(round(fraction * cosmos.shape[1]))
    frame = sim.copy()
    frame[:, :column] = cosmos[:, :column]
    return frame


def command_scenes(number, decision, looks, clip, base, history):
    """One command: looks -> confirm -> instruction -> GR00T acts (simulator, or a wipe into Cosmos) -> check."""
    world = clip.get("world")
    sim_only = world is None
    cosmos = clip.get("cosmos")
    sim = clip["sim"]
    robot = dict(robot_title="ROBOT · NVIDIA GR00T N1.7, pretrained, zero fine-tuning")
    if sim_only:
        robot.update(robot_caption="the actual run · LIBERO simulator", sim_frame=None)
    else:
        robot.update(robot_caption="LIBERO simulator  ·  becomes the NVIDIA Cosmos 3 re-render when GR00T acts",
                     sim_frame=sim[0])
    cosmos_caption = "photoreal re-render by NVIDIA Cosmos 3  ·  motion from the real run (inset)"
    state = dict(base, **robot, command=number, intended=decision["intended"], history=history, robot_frame=sim[0])
    candidate = None
    for look_index, (look, track) in enumerate(looks):
        stage = 0 if candidate is None else 1
        for i in range(max(1, int(round(track.decision_seconds * FPS)))):
            yield MissionState(**state, look=track, look_elapsed=(i + 1) / FPS, live=True, stage=stage,
                               candidate=candidate,
                               banner=f"command {number}: the subject looks at {decision['intended']}  ·  "
                                      f"look {look_index + 1}", banner_color=MUTED)
        event = look["event"]
        text, color = EVENT_TEXT[event]
        color = color or COLORS[look["intent"]]
        candidate = look["intent"] if event in ("candidate", "switch") else (None if event == "cancel" else candidate)
        history.append(f"#{number} saw {look['ground_truth']:<6} → {look['intent']:<6} "
                       f"@{look['decision_seconds']:.1f}s  {event}")
        for _ in range(int(0.9 * FPS)):
            yield MissionState(**state, look=track, look_elapsed=track.decision_seconds, live=False,
                               stage=2 if event == "confirm" else (1 if candidate else 0), candidate=candidate,
                               banner=f"decoded {look['intent']}  →  {text}", banner_color=color)
    confirmed = decision["confirmed"]
    last_track = looks[-1][1]
    settled = dict(state, look=last_track, look_elapsed=last_track.decision_seconds, live=False,
                   confirmed=confirmed, instruction=clip["instruction"])
    for _ in range(int(1.3 * FPS)):
        yield MissionState(**settled, stage=2, banner=f"SELECT {confirmed}  →  “{clip['instruction']}”",
                           banner_color=COLORS[confirmed])
    frames = len(sim)
    label = "" if sim_only else f"{WORLD_TITLES[world]}  ·  NVIDIA Cosmos 3"
    for i in range(int(frames / SOURCE_FPS * FPS)):
        k = min(frames - 1, int(i * SOURCE_FPS / FPS))
        if sim_only:
            extra = dict(robot_frame=sim[k])
        else:
            fraction = min(1.0, i / (WIPE_SECONDS * FPS))
            extra = dict(robot_frame=wiped(cosmos[k], sim[k], fraction), sim_frame=sim[k], wipe=fraction,
                         world_label=label if fraction >= 1 else "", robot_caption=cosmos_caption)
        yield MissionState(**{**settled, **extra}, stage=3, banner="GR00T is acting on the instruction",
                           banner_color=TEXT)
    verdict = ("✓ task complete  ·  LIBERO's own success check", ACCENT) if clip["success"] else \
        ("✗ task failed  ·  LIBERO's own success check", "#ff6b6b")
    history.append(f"#{number} GR00T: {clip['instruction'][:34]} {'✓' if clip['success'] else '✗'}")
    final = dict(robot_frame=sim[-1]) if sim_only else dict(robot_frame=cosmos[-1], sim_frame=sim[-1],
                                                             world_label=label, robot_caption=cosmos_caption)
    for _ in range(int(1.4 * FPS)):
        yield MissionState(**{**settled, **final}, stage=5, banner=verdict[0], banner_color=verdict[1])


def timeline(decisions, tracks, clips, explain, worlds, numbers, subject):
    """Yield MissionState frames in order."""
    stats = {"6-target accuracy, 10 held-out subjects": numbers["accuracy"],
             "wrong actions with two-look confirm": f"{numbers['confirm_wrong']} ({numbers['confirm_detail']})",
             "information rate, early stopping": numbers["bits"],
             "GR00T alone, 5 tasks × 10 episodes": numbers["baseline"]}
    history: list[str] = []
    first = clips[decisions[0]["label"]]
    base = dict(scene="kitchen", commands_total=len(decisions), stats=stats)
    title = ("NEUROPILOT", "Brain signals choose.  A pretrained robot acts.",
             [f"Real recorded EEG (BETA dataset, subject S{subject}) picks one of five kitchen tasks.",
              "NVIDIA GR00T N1.7 carries it out; NVIDIA Cosmos 3 shows the same run in realistic homes.",
              "· every decision, robot motion and number below is replayed from a real run ·"])
    for i in range(int(4.5 * FPS)):
        yield MissionState(**base, robot_frame=first["sim"][0], card=title, fade=min(1.0, (4.5 - i / FPS) / 0.6))

    yield from command_scenes(1, decisions[0], tracks[0], first, base, history)

    sim, edges, cosmos = explain
    for i in range(int(1.2 * FPS)):  # hold on the first frame, then play the run at 0.75x, twice
        yield MissionState(scene="explain", explain_frames=[sim[0], edges[0], cosmos[0]])
    length = min(len(sim), len(edges), len(cosmos))
    for _ in range(2):
        for i in range(int(length / SOURCE_FPS / 0.75 * FPS)):
            k = min(length - 1, int(i * SOURCE_FPS * 0.75 / FPS))
            yield MissionState(scene="explain", explain_frames=[sim[k], edges[k], cosmos[k]])
    for i in range(int(1.0 * FPS)):
        yield MissionState(scene="explain", explain_frames=[sim[-1], edges[-1], cosmos[-1]])

    for number, (decision, looks) in enumerate(zip(decisions[1:], tracks[1:]), start=2):
        yield from command_scenes(number, decision, looks, clips[decision["label"]], base, history)

    hero = worlds["frames"]
    length = min(len(f) for f in hero.values())
    lines = ["Same robot motion in every world.", "edge agreement with the real run, mean of",
             f"all {numbers['cosmos_clips']} Cosmos clips: F1 {numbers['edge_f1']}",
             f"(original simulator: {numbers['edge_f1_sim']})", "success label: from the simulator run"]
    slow = 0.6
    for i in range(int(length / SOURCE_FPS / slow * FPS) + int(0.8 * FPS)):
        k = min(length - 1, int(i * SOURCE_FPS * slow / FPS))
        yield MissionState(scene="worlds", world_frames={w: hero[w][k] for w in WORLD_ORDER},
                           world_titles=WORLD_TITLES, sim_frame=worlds["sim"][k], history=lines)

    end = ("NEUROPILOT", "real EEG  →  decoder  →  GR00T  →  Cosmos",
           [f"Brain → robot: {numbers['bci_correct']} commands chose the intended task, "
            f"{numbers['bci_success']} completed by GR00T",
            f"SSVEP decoder, 6 targets, 10 held-out subjects: {numbers['accuracy']} accuracy  ·  "
            f"{numbers['confirm_wrong']} wrong actions with two-look confirm  ·  {numbers['bits']}",
            f"GR00T N1.7 alone: {numbers['baseline']} kitchen episodes  ·  Cosmos 3: {numbers['cosmos_clips']} "
            f"photoreal clips, edge F1 {numbers['edge_f1']} vs the real run",
            f"· recorded public EEG (subject S{subject}, never used for tuning), replayed offline, not a live participant ·",
            "· success is LIBERO's check in simulation; Cosmos re-renders those runs, GR00T was not re-tested in them ·"])
    last = clips[decisions[-1]["label"]]
    for i in range(int(7 * FPS)):
        yield MissionState(**base, command=len(decisions), robot_frame=last["cosmos"][-1], sim_frame=last["sim"][-1],
                           card=end, fade=min(1.0, (i + 1) / (0.6 * FPS)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("artifacts/videos/step5/neuropilot_mission.mp4"))
    parser.add_argument("--max-frames", type=int, default=0, help="stop early (for a quick preview)")
    args = parser.parse_args()

    jobs = json.loads((GR00T / "bci_jobs.json").read_text())
    decisions = jobs["decisions"]
    for decision, job in zip(decisions, jobs["jobs"]):
        decision["label"] = job["label"]
    numbers = load_numbers()
    config, tracks = build_tracks(decisions, jobs["subject"])

    clips = {}
    for decision in decisions:
        label = decision["label"]
        result = json.loads((GR00T / "remote_results" / label / "result.json").read_text())
        sim, _ = read_frames(COSMOS / "inputs" / label / "rgb.mp4")
        clips[label] = {"sim": sim, "instruction": result["instruction"], "success": result["successes"][0]}
        world = COMMAND_WORLDS.get(label)
        if world:
            clips[label].update(world=world, cosmos=read_frames(cosmos_clip(label, world))[0])
    first = decisions[0]["label"]
    explain = (clips[first]["sim"], read_frames(COSMOS / "inputs" / first / "control_edge.mp4")[0],
               read_frames(cosmos_clip(first, "modern_kitchen"))[0])
    hero = "bci_02_PLATE"
    worlds = {"frames": {w: read_frames(cosmos_clip(hero, w))[0] for w in WORLD_ORDER}, "sim": clips[hero]["sim"]}

    renderer = MissionRenderer(config.targets, list(config.channels), config.margin_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080",
                                "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                                "-pix_fmt", "yuv420p", str(args.output)], stdin=subprocess.PIPE)
    count = 0
    for state in timeline(decisions, tracks, clips, explain, worlds, numbers, jobs["subject"]):
        encoder.stdin.write(renderer.frame(state).tobytes())
        count += 1
        if count % 250 == 0:
            print(f"{count} frames ({count / FPS:.0f} s)", flush=True)
        if args.max_frames and count >= args.max_frames:
            break
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    renderer.close()
    print(f"wrote {args.output}: {count} frames, {count / FPS:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
