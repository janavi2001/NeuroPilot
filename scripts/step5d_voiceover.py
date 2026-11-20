#!/usr/bin/env python3
"""Generate the Step 5d voiceover with ElevenLabs, one file per script segment, with word timings.

Uses the text-to-speech "with-timestamps" endpoint, which returns the audio and
per-character start/end times; captions and highlights in the video are synced
to those. The API key is read from ~/.config/elevenlabs_api_key (or
ELEVENLABS_API_KEY) and never printed. A segment is re-generated only when its
spoken text, voice or settings change, so edits cost only the changed lines.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

SCRIPT = Path("scripts/step5d_narration.yaml")
OUT_DIR = Path("artifacts/step5_voice")
KEY_FILE = Path.home() / ".config" / "elevenlabs_api_key"
VOICE_SETTINGS = {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0, "use_speaker_boost": True}


def api_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY") or (KEY_FILE.read_text().strip() if KEY_FILE.is_file() else "")
    if not key:
        raise SystemExit(f"No ElevenLabs key: set ELEVENLABS_API_KEY or write it to {KEY_FILE}")
    return key


def load_script(path: Path = SCRIPT) -> tuple[dict, list[dict]]:
    raw = yaml.safe_load(path.read_text())
    segments = raw["segments"]
    ids = [s["id"] for s in segments]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate segment ids")
    return raw["voice"], segments


def fingerprint(voice: dict, segment: dict, previous: str, following: str) -> str:
    payload = json.dumps([voice, segment.get("speak", segment["text"]), VOICE_SETTINGS, previous, following])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def synthesize(key: str, voice: dict, text: str, previous: str, following: str) -> dict:
    body = {"text": text, "model_id": voice["model"], "voice_settings": VOICE_SETTINGS}
    if previous:
        body["previous_text"] = previous  # keeps intonation continuous across segments
    if following:
        body["next_text"] = following
    request = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice['id']}/with-timestamps?output_format=mp3_44100_128",
        data=json.dumps(body).encode(), method="POST",
        headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise SystemExit(f"ElevenLabs HTTP {error.code}: {error.read()[:300].decode(errors='replace')}") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--script", type=Path, default=SCRIPT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--only", nargs="*", help="segment ids to (re)generate; default all that changed")
    args = parser.parse_args()

    voice, segments = load_script(args.script)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    spoken = [s.get("speak", s["text"]) for s in segments]
    key = None
    billed = 0
    for index, segment in enumerate(segments):
        previous = spoken[index - 1] if index else ""
        following = spoken[index + 1] if index + 1 < len(segments) else ""
        stamp = fingerprint(voice, segment, previous, following)
        meta_path = args.out_dir / f"{segment['id']}.json"
        if meta_path.is_file() and json.loads(meta_path.read_text()).get("fingerprint") == stamp \
                and (args.only is None or segment["id"] not in args.only):
            print(f"cached  {segment['id']}")
            continue
        if args.only is not None and segment["id"] not in args.only:
            continue
        key = key or api_key()
        result = synthesize(key, voice, spoken[index], previous, following)
        (args.out_dir / f"{segment['id']}.mp3").write_bytes(base64.b64decode(result["audio_base64"]))
        alignment = result.get("alignment") or result.get("normalized_alignment")
        meta_path.write_text(json.dumps({"id": segment["id"], "text": segment["text"], "spoken": spoken[index],
                                         "voice": voice, "fingerprint": stamp, "alignment": alignment}, indent=1))
        billed += len(spoken[index])
        end = alignment["character_end_times_seconds"][-1] if alignment else float("nan")
        print(f"made    {segment['id']}  {len(spoken[index])} chars  {end:.1f} s")
    print(f"characters billed this run: {billed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
