#!/usr/bin/env python3
"""GPU side of Step 5b: Cosmos 3 Nano edge transfer for each job, in priority order.

Follows NVIDIA's Diffusers transfer runner (cosmos3-inference-transfer,
references/diffusers.md § 7): Cosmos3OmniModularPipeline, UniPC scheduler with
the spec's flow shift, the edge spec's sampling values, the shared negative
prompt, and the safety checker disabled for edge as in the recipe. The
pipeline is loaded once. Each finished job writes vision.mp4 and result.json
right away; finished jobs are skipped on a re-run. Writes RUN_DONE at the end.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Sampling values of the recipe's specs/edge.json.
EDGE_SPEC = {"shift": 10.0, "num_steps": 50, "guidance": 3.0, "control_guidance": 1.5, "num_conditional_frames": 1}


def log(path: Path, message: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with path.open("a") as handle:
        handle.write(line + "\n")


def compact(path: Path) -> str:
    return json.dumps(json.loads(path.read_text()), ensure_ascii=True, separators=(",", ":"))


def load_pipe():
    import torch
    from diffusers import Cosmos3OmniModularPipeline

    start = time.time()
    pipe = Cosmos3OmniModularPipeline.from_pretrained("nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16)
    pipe.load_components(torch_dtype=torch.bfloat16)
    # load_components logs a failed component and leaves it None (2026-11-19: HF_HUB_OFFLINE=1
    # left the transformer unloaded and the first call failed on `None.dtype`); fail here instead.
    missing = [name for name in ("transformer", "vae", "text_tokenizer") if getattr(pipe, name, None) is None]
    if missing:
        raise RuntimeError(f"pipeline components not loaded: {missing}")
    pipe.to("cuda")
    pipe.disable_safety_checker()  # the recipe disables it for edge and depth controls
    return pipe, round(time.time() - start, 1)


def run_job(pipe, job: dict, inputs: Path, negative: str, out_dir: Path) -> dict:
    import torch
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video, load_video

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=EDGE_SPEC["shift"],
                                                         use_karras_sigmas=False)
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    videos = pipe(
        prompt=compact(inputs / job["prompt"]),
        negative_prompt=negative,
        control_videos={"edge": load_video(str(inputs / job["control"]))},
        control_guidance=EDGE_SPEC["control_guidance"],
        num_frames=job["num_frames"],
        num_video_frames_per_chunk=job["num_frames"],
        num_conditional_frames=EDGE_SPEC["num_conditional_frames"],
        height=job["size"],
        width=job["size"],
        fps=float(job["fps"]),
        num_inference_steps=EDGE_SPEC["num_steps"],
        guidance_scale=EDGE_SPEC["guidance"],
        generator=torch.Generator(device="cuda").manual_seed(job["seed"]),
        output="videos",
    )
    seconds = round(time.time() - start, 1)
    output = out_dir / job["name"] / "vision.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(videos, str(output), fps=job["fps"], macro_block_size=1)
    return {**job, "status": "ok", "generation_seconds": seconds, "video": str(output.relative_to(out_dir)),
            "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 1),
            "sampling": EDGE_SPEC, "model": "nvidia/Cosmos3-Nano", "precision": "bf16",
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("/root/cosmos_inputs"))
    parser.add_argument("--jobs", type=Path, default=None, help="defaults to <inputs>/jobs.json")
    parser.add_argument("--negative-prompt", type=Path, required=True, help="the recipe's assets/negative_prompt.json")
    parser.add_argument("--out-dir", type=Path, default=Path("/root/cosmos_results"))
    parser.add_argument("--limit", type=int, default=0, help="run at most this many pending jobs (0 = all)")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--run-name", default="run", help="progress log is <out-dir>/<run-name>.progress.log")
    args = parser.parse_args()

    jobs = json.loads((args.jobs or args.inputs / "jobs.json").read_text())["jobs"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress = args.out_dir / f"{args.run_name}.progress.log"
    pending = [job for job in jobs
               if not (args.out_dir / job["name"] / "result.json").is_file()
               or json.loads((args.out_dir / job["name"] / "result.json").read_text()).get("status") != "ok"]
    if args.limit:
        pending = pending[:args.limit]
    log(progress, f"{len(jobs)} jobs, running {len(pending)} pending (limit {args.limit or 'none'})")
    negative = compact(args.negative_prompt)
    pipe, load_seconds = load_pipe()
    log(progress, f"pipeline loaded in {load_seconds}s")
    for job in pending:
        result_path = args.out_dir / job["name"] / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, args.retries + 2):
            log(progress, f"start {job['name']} attempt {attempt}: {job['num_frames']} frames")
            try:
                result = run_job(pipe, job, args.inputs, negative, args.out_dir)
                break
            except Exception as error:  # recorded and retried, never hidden
                log(progress, f"error {job['name']} attempt {attempt}: {error!r}")
                result = {**job, "status": "error", "error": repr(error), "traceback": traceback.format_exc()}
        result_path.write_text(json.dumps(result, indent=2))
        log(progress, f"done {job['name']}: status={result['status']} "
                      f"seconds={result.get('generation_seconds')} peak={result.get('peak_gpu_memory_gib')}GiB")
    with progress.open("a") as handle:
        handle.write("RUN_DONE\n")
    print("RUN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
