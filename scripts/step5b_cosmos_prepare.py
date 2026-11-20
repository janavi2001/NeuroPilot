#!/usr/bin/env python3
"""Step 5b, local: build Cosmos 3 transfer inputs from the brain-chosen GR00T episodes.

For each Step 5a BCI episode: crop the third-person view, upscale to 640x640,
compute NVIDIA's medium Canny edge control, and write one structured prompt per
world. Jobs are ordered by priority so a budget cut still leaves the most
useful clips: (1) every episode in the photoreal kitchen, (2) one hero episode
in every other world, (3) the remaining episode/world pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cosmos.controls import build_control  # noqa: E402
from cosmos.scenarios import WORLDS, build_prompt  # noqa: E402

GR00T_RESULTS = Path("artifacts/step5_gr00t/remote_results")
OUT_DIR = Path("artifacts/step5_cosmos/inputs")
FIRST_WORLD = "modern_kitchen"


def ordered_jobs(clips: list[str], worlds: list[str], hero: str) -> list[tuple[str, str]]:
    first, others = worlds[0], worlds[1:]
    jobs = [(clip, first) for clip in clips]
    jobs += [(hero, world) for world in others]
    jobs += [(clip, world) for world in others for clip in clips if clip != hero]
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=GR00T_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--hero", default="bci_02_PLATE", help="episode shown in every world first")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    episodes = {}
    for result_path in sorted(args.results.glob("bci_*/result.json")):
        result = json.loads(result_path.read_text())
        if result.get("status") != "ok" or not result.get("videos"):
            raise SystemExit(f"{result_path} has no successful video")
        episodes[result["label"]] = result
    if args.hero not in episodes:
        raise SystemExit(f"hero {args.hero} not among {sorted(episodes)}")

    worlds = [FIRST_WORLD] + [key for key in WORLDS if key != FIRST_WORLD]
    clips = {}
    for label, result in episodes.items():
        meta = build_control(args.results / result["videos"][0], args.out_dir / label)
        meta.update(label=label, instruction=result["instruction"], gr00t_success=result["successes"][0],
                    intended=result["intended"], object=result["object"])
        (args.out_dir / label / "meta.json").write_text(json.dumps(meta, indent=2))
        clips[label] = meta
        print(f"{label:<14} {meta['source_frames']} -> {meta['num_frames']} frames @ {meta['fps']:g} fps, "
              f"edges {meta['edge_pixel_fraction']:.1%} of pixels")
        for key in worlds:
            prompt = build_prompt(meta["instruction"], WORLDS[key], meta["num_frames"], meta["fps"])
            (args.out_dir / label / f"prompt_{key}.json").write_text(json.dumps(prompt, indent=2))

    jobs = [{"name": f"{clip}__{world}", "clip": clip, "world": world, "world_title": WORLDS[world].title,
             "instruction": clips[clip]["instruction"], "num_frames": clips[clip]["num_frames"],
             "fps": clips[clip]["fps"], "size": clips[clip]["size"], "seed": args.seed,
             "control": f"{clip}/control_edge.mp4", "prompt": f"{clip}/prompt_{world}.json"}
            for clip, world in ordered_jobs(list(clips), worlds, args.hero)]
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "jobs": jobs}
    (args.out_dir / "jobs.json").write_text(json.dumps(payload, indent=2))
    for priority, job in enumerate(jobs, start=1):
        print(f"{priority:>2}. {job['name']}")
    print(f"wrote {args.out_dir / 'jobs.json'} ({len(jobs)} jobs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
