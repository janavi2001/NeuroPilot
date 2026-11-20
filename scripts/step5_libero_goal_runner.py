#!/usr/bin/env python3
"""GPU side of Step 5: run official GR00T-N1.7 rollouts in LIBERO-Goal for a jobs file.

Run with the LIBERO venv python from the Isaac-GR00T checkout, with the policy
server already listening. Each job's result is written as soon as it finishes,
and finished jobs are skipped on a re-run, so a retry resumes where it stopped.
A job that raises is retried up to --retries times, then recorded as an error.
Writes RUN_DONE to the progress log when every job has a result.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def log(path: Path, message: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with path.open("a") as handle:
        handle.write(line + "\n")


def check_env_names(jobs: list[dict]) -> None:
    import gymnasium as gym
    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

    register_libero_envs()
    missing = sorted({job["env_name"] for job in jobs} - set(gym.registry))
    if missing:
        raise SystemExit(f"Unknown LIBERO env names: {missing}")


def run_job(job: dict, out_dir: Path, args) -> dict:
    from gr00t.eval.rollout_policy import run_gr00t_sim_policy

    video_dir = out_dir / job["label"] / "videos"
    start = time.time()
    _, successes, infos = run_gr00t_sim_policy(
        env_name=job["env_name"],
        n_episodes=job["episodes"],
        max_episode_steps=args.max_episode_steps,
        policy_client_host=args.host,
        policy_client_port=args.port,
        n_envs=1,
        n_action_steps=8,
        video_dir=str(video_dir),
        seed=job["seed"],
    )
    successes = [bool(s) for s in successes]
    return {
        **job,
        "status": "ok",
        "successes": successes,
        "success_rate": sum(successes) / len(successes) if successes else None,
        "episode_lengths": [int(n) for n in infos.get("episode_lengths", [])],
        "seconds": round(time.time() - start, 1),
        "videos": sorted(str(p.relative_to(out_dir)) for p in video_dir.rglob("*.mp4")),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("/root/neuropilot_results"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    jobs = json.loads(args.jobs.read_text())["jobs"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress = args.out_dir / f"{args.jobs.stem}.progress.log"
    check_env_names(jobs)
    for job in jobs:
        result_path = args.out_dir / job["label"] / "result.json"
        if result_path.is_file() and json.loads(result_path.read_text()).get("status") == "ok":
            log(progress, f"skip {job['label']} (already done)")
            continue
        result_path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, args.retries + 2):
            log(progress, f"start {job['label']} attempt {attempt}: {job['env_name']} x{job['episodes']}")
            try:
                result = run_job(job, args.out_dir, args)
                break
            except Exception as error:  # recorded and retried, never hidden
                log(progress, f"error {job['label']} attempt {attempt}: {error!r}")
                result = {**job, "status": "error", "error": repr(error), "traceback": traceback.format_exc()}
        result_path.write_text(json.dumps(result, indent=2))
        log(progress, f"done {job['label']}: status={result['status']} successes={result.get('successes')}")
    with progress.open("a") as handle:
        handle.write("RUN_DONE\n")  # bare marker line for step5_wait_remote.py
    print("RUN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
