#!/usr/bin/env python3
"""Step 5b, local: measure every Cosmos clip and build comparison videos.

For each generated clip: edge agreement between the Cosmos output and the
control edges of the real GR00T episode (and, as an upper bound, between the
original simulator frames and the same control). Writes summary.json, a
side-by-side video per clip, and a 2x2 world grid for the hero episode.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cosmos.controls import edge_agreement, read_frames  # noqa: E402

ROOT = Path("artifacts/step5_cosmos")


def control_edges(path: Path):
    frames, _ = read_frames(path)
    return ((frames > 127) * 255).astype("uint8")


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--hero", default="bci_02_PLATE")
    args = parser.parse_args()
    inputs, results, out = args.root / "inputs", args.root / "remote_results", args.root / "compare"
    out.mkdir(parents=True, exist_ok=True)

    clips = []
    for result_path in sorted(results.glob("*/result.json")):
        result = json.loads(result_path.read_text())
        if result.get("status") != "ok":
            clips.append({"name": result_path.parent.name, "status": result.get("status"), "error": result.get("error")})
            continue
        meta = json.loads((inputs / result["clip"] / "meta.json").read_text())
        control = control_edges(inputs / result["control"])
        generated, _ = read_frames(results / result["video"])
        original, _ = read_frames(inputs / result["clip"] / "rgb.mp4")
        clips.append({
            "name": result["name"], "status": "ok", "clip": result["clip"], "world": result["world"],
            "instruction": result["instruction"], "intended": meta["intended"], "gr00t_success": meta["gr00t_success"],
            "frames": result["num_frames"], "fps": result["fps"], "seconds_generated": result["generation_seconds"],
            "peak_gpu_memory_gib": result["peak_gpu_memory_gib"], "seed": result["seed"],
            "edge_agreement": edge_agreement(control, generated),
            "edge_agreement_original_sim": edge_agreement(control, original),
        })
        ffmpeg("-i", str(inputs / result["clip"] / "rgb.mp4"), "-i", str(results / result["video"]),
               "-filter_complex", "[0:v][1:v]hstack", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
               str(out / f"{result['name']}.mp4"))
        print(f"{result['name']:<32} F1 {clips[-1]['edge_agreement']['f1']:.3f} "
              f"(sim {clips[-1]['edge_agreement_original_sim']['f1']:.3f})  {result['generation_seconds']:.0f}s")

    hero = {c["world"]: results / f"{c['name']}/vision.mp4" for c in clips
            if c.get("status") == "ok" and c["clip"] == args.hero}
    order = ["modern_kitchen", "warehouse", "office", "night_kitchen"]
    if all(world in hero for world in order):
        inputs_args = [arg for world in order for arg in ("-i", str(hero[world]))]
        ffmpeg(*inputs_args, "-filter_complex", "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack",
               "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out / f"{args.hero}__4_worlds.mp4"))
        print(f"wrote {out / f'{args.hero}__4_worlds.mp4'}")

    ok = [c for c in clips if c.get("status") == "ok"]
    summary = {
        "model": "nvidia/Cosmos3-Nano (bf16, Diffusers Cosmos3OmniModularPipeline, edge control)",
        "clips_ok": len(ok), "clips_failed": len(clips) - len(ok),
        "mean_edge_f1": round(sum(c["edge_agreement"]["f1"] for c in ok) / len(ok), 4),
        "mean_edge_f1_original_sim": round(sum(c["edge_agreement_original_sim"]["f1"] for c in ok) / len(ok), 4),
        "clips": clips,
    }
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{len(ok)} ok, {len(clips) - len(ok)} failed; mean edge F1 {summary['mean_edge_f1']:.3f} "
          f"(original sim {summary['mean_edge_f1_original_sim']:.3f}); wrote {args.root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
