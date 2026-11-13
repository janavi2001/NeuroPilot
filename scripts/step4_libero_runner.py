#!/usr/bin/env python3
"""Run official GR00T closed-loop LIBERO with a supplied instruction."""

from __future__ import annotations

import argparse
import os

import gymnasium as gym
from libero.libero import benchmark


def register_instruction_task(task_name: str, instruction: str) -> str:
    from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv, get_libero_path

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = None
    for task_id in range(suite.get_num_tasks()):
        candidate = suite.get_task(task_id)
        if candidate.name == task_name:
            task = candidate
            break
    if task is None:
        raise ValueError(f"Unknown LIBERO object task: {task_name}")

    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_name = "libero_sim/neuropilot_bci_task"
    gym.register(
        id=env_name,
        entry_point=LiberoEnv,
        kwargs={"task_bddl_file": bddl_file, "task_description": instruction},
    )
    return env_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--task-name", default="pick_up_the_alphabet_soup_and_place_it_in_the_basket")
    parser.add_argument("--policy-client-host", default="127.0.0.1")
    parser.add_argument("--policy-client-port", type=int, default=5555)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-dir", default="/tmp/neuropilot_step4_videos")
    args = parser.parse_args()

    from gr00t.eval.rollout_policy import run_gr00t_sim_policy

    env_name = register_instruction_task(args.task_name, args.instruction)
    print(f"NeuroPilot instruction: {args.instruction}")
    print(f"LIBERO physical task: {args.task_name}")
    print(f"GR00T policy server: {args.policy_client_host}:{args.policy_client_port}")
    result = run_gr00t_sim_policy(
        env_name=env_name,
        n_episodes=args.episodes,
        max_episode_steps=720,
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
        n_envs=1,
        n_action_steps=8,
        video_dir=args.video_dir,
        seed=args.seed,
    )
    _, successes, infos = result
    success_rate = sum(bool(item) for item in successes) / max(1, len(successes))
    print(f"GR00T/LIBERO results: successes={successes}")
    print(f"GR00T/LIBERO success rate: {success_rate:.3f}")
    print(f"GR00T/LIBERO episode lengths: {infos.get('episode_lengths')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
