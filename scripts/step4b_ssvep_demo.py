#!/usr/bin/env python3
"""Step 4b: real BETA SSVEP EEG -> 4-target selection -> scripted MuJoCo pick."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from bci.ssvep_data import load_beta_trials
from bci.ssvep_decoder import UNSURE, SSVEPDecoder
from interface.ssvep_selector import DEFAULT_CONFIG, SSVEPSelector, load_ssvep_config
from robot.scripted_controller import ScriptedRobotController
from simulation.environment import RobotSimulation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--subject", type=int, default=1, help="BETA subject to replay (1-70)")
    parser.add_argument(
        "--sequence", nargs="+", default=["PHONE", "CANCEL", "CUP"],
        help="Targets the subject was really looking at; each uses that target's next recorded trial",
    )
    parser.add_argument("--log", type=Path, default=Path("artifacts/logs/step4b_ssvep_demo.jsonl"))
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
    decoder = SSVEPDecoder(
        config.targets, len(config.channels), metadata["sampling_rate_hz"],
        config.window_seconds, config.harmonics, config.margin_threshold,
    )
    print(f"Dataset: {metadata['dataset']} subject S{args.subject}; {metadata['trials']} recorded trials for the 4 targets")
    print(f"Window: {config.window_seconds:.1f} s from stimulus onset; channels {','.join(metadata['channels'])}")
    print("Targets: " + ", ".join(f"{name}={frequency:.1f} Hz" for name, frequency in config.targets.items()))
    print(f"UNSURE margin threshold: {config.margin_threshold:.3f} (raw CCA margin, not a probability)")

    print("\nAll recorded trials for this subject:")
    correct = unsure = 0
    for index, (window, truth) in enumerate(zip(eeg, truths), start=1):
        prediction = decoder.predict(window)
        correct += prediction["best_target"] == truth
        unsure += not prediction["accepted"]
        print(f"  trial {index:2d}: looked at {truth:<6} -> {prediction['intent']:<6} "
              f"(best {prediction['best_target']}, score {prediction['score']:.3f}, margin {prediction['margin']:.3f})")
    print(f"Subject accuracy before UNSURE: {correct}/{len(truths)} = {correct / len(truths):.1%}; UNSURE: {unsure}/{len(truths)}")

    print(f"\nSelection replay, intended sequence {' -> '.join(sequence)}:")
    selector = SSVEPSelector(config.objects, config.cancel_target)
    used: set[int] = set()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a") as log:
        for intended in sequence:
            candidates = [index for index, truth in enumerate(truths) if truth == intended and index not in used]
            prediction = None
            for index in candidates:
                used.add(index)
                start = time.perf_counter()
                prediction = decoder.predict(eeg[index])
                latency_ms = 1000 * (time.perf_counter() - start)
                state = selector.apply(prediction)
                print(f"  looked at {intended:<6} (trial {index + 1}) -> {prediction['intent']:<6} "
                      f"margin {prediction['margin']:.3f} -> selected={state.selected_object}")
                log.write(json.dumps({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "decoder": "CCA", "dataset": "BETA", "subject": args.subject, "trial": index + 1,
                    "ground_truth": intended, "prediction": prediction["intent"], "score": prediction["score"],
                    "margin": prediction["margin"], "selected_object": state.selected_object, "latency_ms": latency_ms,
                }) + "\n")
                if prediction["intent"] != UNSURE:
                    break
            if prediction is None or prediction["intent"] == UNSURE:
                print(f"  no confident decision for {intended}; the selection is unchanged")

        target = selector.state.selected_object
        intended_final = None
        for name in sequence:
            intended_final = None if name == config.cancel_target else name
        print(f"History: {' | '.join(selector.state.history)}")
        print(f"Final selection: {target}  (intended: {intended_final})")
        if target is None:
            print("No object selected; no robot action issued.")
            return 0
        if target != intended_final:
            print("WARNING: the decoded selection differs from what the subject was looking at.")

        simulation = RobotSimulation(gui=args.gui)
        try:
            start = time.perf_counter()
            result = ScriptedRobotController(simulation).execute(action="pick", target=target)
            robot_seconds = time.perf_counter() - start
            print(f"Robot action: pick {target}")
            print(f"Grasp: {'SUCCESS' if result['grasp_success'] else 'FAIL'}")
            print(f"Object height: {result['initial_z']:.3f} -> {result['final_z']:.3f} (delta {result['lift_delta']:.3f})")
            print(f"Object lifted: {'YES' if result['lifted'] else 'NO'}")
            target_correct = target == intended_final
            log.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(), "selected_object": target,
                "intended_object": intended_final, "robot_action": f"pick {target}", "lifted": bool(result["lifted"]),
                "target_correct": target_correct, "success": bool(result["lifted"]) and target_correct,
                "lift_delta": result["lift_delta"], "robot_seconds": robot_seconds,
            }) + "\n")
            if not result["lifted"]:
                raise RuntimeError("Programmatic lift verification failed")
            if not target_correct:
                print(f"SSVEP -> ROBOT TEST: FAIL (decoder selected {target}; subject looked at {intended_final}; robot lifted {target})")
                return 1
            print("SSVEP -> ROBOT TEST: PASS")
        finally:
            if args.gui:
                input("Press Enter to close the MuJoCo window...")
            simulation.close()
    print(f"Log: {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
