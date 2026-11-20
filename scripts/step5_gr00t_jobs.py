#!/usr/bin/env python3
"""Step 5 jobs for GR00T in the LIBERO-Goal kitchen, built locally.

  baseline   every kitchen task, N episodes each (GR00T alone, no BCI)
  bci        real BETA EEG replay -> early-stopped CCA -> two-look confirm ->
             task instruction; one job per confirmed command

Each kitchen object maps to an exact LIBERO-Goal instruction
(interface/ssvep_targets_kitchen.yaml). Looks are decoded from recorded trials
chosen by ground truth, never by whether they decode correctly; a wrong
confirmation becomes a wrong job and is reported as such.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interface.ssvep_confirm import ConfirmingSelector  # noqa: E402

KITCHEN_CONFIG = Path("interface/ssvep_targets_kitchen.yaml")
OUT_DIR = Path("artifacts/step5_gr00t")


def load_tasks(path: Path = KITCHEN_CONFIG) -> dict[str, str]:
    """Object name -> LIBERO-Goal instruction."""
    return {str(name).upper(): str(text) for name, text in yaml.safe_load(Path(path).read_text())["tasks"].items()}


def libero_env_name(instruction: str) -> str:
    """LIBERO-Goal task names are the instruction with spaces as underscores."""
    return "libero_sim/" + instruction.strip().lower().replace(" ", "_")


def baseline_jobs(tasks: dict[str, str], episodes: int, seed: int) -> list[dict[str, Any]]:
    return [{"label": f"baseline_{name}", "object": name, "instruction": text, "env_name": libero_env_name(text),
             "episodes": episodes, "seed": seed} for name, text in tasks.items()]


def confirm_command(intended: str, next_look: Callable[[str], dict | None], selector: ConfirmingSelector,
                    max_looks: int) -> tuple[list[dict], str | None]:
    """Look at ``intended`` until a command is confirmed, trials run out, or ``max_looks``.

    Returns every look and the confirmed object (which may differ from intended),
    or None if nothing was confirmed: the robot is then left idle.
    """
    selector.reset()
    looks = []
    for _ in range(max_looks):
        prediction = next_look(intended)
        if prediction is None:
            break
        state = selector.apply(prediction)
        looks.append({**prediction, "event": state.event})
        if state.confirmed is not None:
            return looks, state.confirmed
    return looks, None


def bci_jobs(subject: int, config_path: Path, sequence: list[str], max_looks: int,
             seed: int) -> tuple[list[dict], list[dict]]:
    from bci.ssvep_data import load_beta_trials
    from bci.ssvep_stopping import EarlyStoppingDecoder, StoppingRule
    from interface.ssvep_selector import load_ssvep_config

    config = load_ssvep_config(config_path)
    tasks = load_tasks(config_path)
    eeg, labels, metadata = load_beta_trials(subject, config.channels, config.window_seconds,
                                             list(config.targets.values()))
    name_of = {frequency: name for name, frequency in config.targets.items()}
    unused: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        unused.setdefault(name_of[float(label)], []).append(index)
    decoder = EarlyStoppingDecoder(config.targets, len(config.channels), StoppingRule.from_yaml(config_path),
                                   metadata["sampling_rate_hz"], config.window_seconds, config.harmonics,
                                   config.margin_threshold)

    def next_look(intended: str) -> dict | None:
        if not unused.get(intended):
            return None
        index = unused[intended].pop(0)
        result = decoder.predict(eeg[index])
        return {"trial": index + 1, "ground_truth": intended, "intent": result["intent"],
                "score": round(result["score"], 4), "margin": round(result["margin"], 4),
                "decision_seconds": result["decision_seconds"]}

    selector = ConfirmingSelector(config.objects, config.cancel_target)
    decisions, jobs = [], []
    for number, intended in enumerate(sequence, start=1):
        looks, confirmed = confirm_command(intended, next_look, selector, max_looks)
        decision = {"command": number, "intended": intended, "looks": looks, "confirmed": confirmed,
                    "correct": confirmed == intended,
                    "brain_seconds": round(sum(look["decision_seconds"] for look in looks), 2)}
        decisions.append(decision)
        if confirmed is not None:
            text = tasks[confirmed]
            jobs.append({"label": f"bci_{number:02d}_{confirmed}", "object": confirmed, "intended": intended,
                         "instruction": text, "env_name": libero_env_name(text), "episodes": 1,
                         "seed": seed + number})
    return decisions, jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    base = sub.add_parser("baseline")
    base.add_argument("--episodes", type=int, default=10)
    base.add_argument("--seed", type=int, default=0)
    bci = sub.add_parser("bci")
    bci.add_argument("--subject", type=int, default=1, help="held-out BETA subject (S1-S10)")
    bci.add_argument("--sequence", nargs="+", default=None, help="objects in order; default all five")
    bci.add_argument("--max-looks", type=int, default=4)
    bci.add_argument("--seed", type=int, default=100)
    for p in (base, bci):
        p.add_argument("--config", type=Path, default=KITCHEN_CONFIG)
        p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tasks = load_tasks(args.config)
    if args.mode == "baseline":
        jobs = baseline_jobs(tasks, args.episodes, args.seed)
        payload = {"created_utc": created, "mode": "baseline", "jobs": jobs}
    else:
        sequence = [name.upper() for name in (args.sequence or list(tasks))]
        unknown = set(sequence) - set(tasks)
        if unknown:
            raise SystemExit(f"Unknown objects {sorted(unknown)}; choose from {list(tasks)}")
        decisions, jobs = bci_jobs(args.subject, args.config, sequence, args.max_looks, args.seed)
        payload = {"created_utc": created, "mode": "bci", "subject": args.subject, "decisions": decisions,
                   "jobs": jobs}
        for d in decisions:
            looks = ", ".join(f"{look['intent']}@{look['decision_seconds']:.1f}s" for look in d["looks"])
            verdict = "no command" if d["confirmed"] is None else ("OK" if d["correct"] else "WRONG")
            print(f"#{d['command']} looked at {d['intended']:<6} [{looks}] -> {d['confirmed']} ({verdict})")
    path = args.out_dir / f"{args.mode}_jobs.json"
    path.write_text(json.dumps(payload, indent=2))
    for job in jobs:
        print(f"{job['label']:<20} {job['env_name']}  x{job['episodes']}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
