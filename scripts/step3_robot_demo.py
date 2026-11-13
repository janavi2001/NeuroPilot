#!/usr/bin/env python3
"""Step 3: real EEG -> object selection -> scripted MuJoCo pick."""

from __future__ import annotations

import argparse
from pathlib import Path

from bci.decoder import BCIDecoder
from bci.preprocessing import load_real_trials
from interface.replay import RecordedEEGReplay
from interface.selector import ObjectSelector
from robot.scripted_controller import ScriptedRobotController
from simulation.environment import RobotSimulation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg", type=Path, default=Path("data/raw/MI_BCIC_IV2a_mat/A01T.mat"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt"))
    parser.add_argument("--start-trial", type=int, default=1)
    parser.add_argument("--limit", type=int, default=2, help="Default trials 1-2 decode RIGHT then LEFT, selecting CUP")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gui", action="store_true", help="Open the MuJoCo visualization window")
    args = parser.parse_args()
    if args.start_trial < 1 or args.limit < 1:
        parser.error("--start-trial and --limit must be positive")

    end = args.start_trial - 1 + args.limit
    eeg, labels, _ = load_real_trials(args.eeg, end)
    eeg = eeg[args.start_trial - 1:end]
    labels = labels[args.start_trial - 1:end]
    decoder = BCIDecoder(args.checkpoint, device=args.device)
    selector = ObjectSelector()
    events = list(RecordedEEGReplay(eeg, labels, decoder))
    for event in events:
        state = selector.apply(str(event.prediction["intent"]))
        print(f"Trial {event.trial_number}: {event.prediction['intent']} -> current={state.current_object} selected={state.selected_object}")
    target = selector.state.selected_object
    if target is None:
        raise RuntimeError("BCI replay did not select an object; no robot action will be issued")
    print(f"BCI intent: SELECT {target}")

    simulation = RobotSimulation(gui=args.gui)
    try:
        controller = ScriptedRobotController(simulation)
        result = controller.execute(action="pick", target=target)
        print(f"Robot target: {target}")
        print(f"Grasp: {'SUCCESS' if result['grasp_success'] else 'FAIL'}")
        print(f"Object height: {result['initial_z']:.3f} -> {result['final_z']:.3f} (delta {result['lift_delta']:.3f})")
        print(f"Object lifted: {'YES' if result['lifted'] else 'NO'}")
        if not result["lifted"]:
            raise RuntimeError("Programmatic lift verification failed")
        print("BRAIN -> ROBOT TEST: PASS")
    finally:
        if args.gui:
            input("Press Enter to close the MuJoCo window...")
        simulation.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
