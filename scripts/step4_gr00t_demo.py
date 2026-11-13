#!/usr/bin/env python3
"""Step 4: real EEG -> intent -> official GR00T/LIBERO on a GPU host."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from bci.decoder import BCIDecoder
from bci.preprocessing import load_real_trials
from interface.intent import IntentAdapter
from interface.replay import RecordedEEGReplay
from interface.selector import ObjectSelector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg", type=Path, default=Path("data/raw/MI_BCIC_IV2a_mat/A01T.mat"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt"))
    parser.add_argument("--ssh-host", required=True, help="GPU host, e.g. root@ssh1.vast.ai")
    parser.add_argument("--ssh-port", type=int, default=27750)
    parser.add_argument("--remote-repo", default="/root/Isaac-GR00T")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    eeg, labels, _ = load_real_trials(args.eeg, 2)
    decoder = BCIDecoder(args.checkpoint, device=args.device)
    selector = ObjectSelector()
    for event in RecordedEEGReplay(eeg, labels, decoder):
        state = selector.apply(str(event.prediction["intent"]))
        print(
            f"Trial {event.trial_number}: {event.prediction['intent']} "
            f"-> current={state.current_object} selected={state.selected_object}"
        )
    target = selector.state.selected_object
    if target is None:
        raise RuntimeError("BCI replay did not select an object")

    adapter = IntentAdapter()
    instruction = adapter.convert(target)
    libero_instruction = adapter.convert_libero(target)
    print(f"BCI intent: SELECT {target}")
    print(f"Semantic instruction: {instruction}")
    print(f"LIBERO task adapter instruction: {libero_instruction}")

    remote_python = f"{args.remote_repo}/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
    remote_runner = f"{args.remote_repo}/scripts/step4_libero_runner.py"
    remote = (
        f"cd {shlex.quote(args.remote_repo)} && "
        f"{shlex.quote(remote_python)} {shlex.quote(remote_runner)} "
        f"--instruction {shlex.quote(libero_instruction)} --episodes {args.episodes}"
    )
    print("Launching official GR00T/LIBERO rollout on the GPU host...")
    completed = subprocess.run(
        ["ssh", "-p", str(args.ssh_port), args.ssh_host, remote],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Remote GR00T/LIBERO rollout failed with code {completed.returncode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
