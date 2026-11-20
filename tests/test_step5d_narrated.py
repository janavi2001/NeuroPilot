import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from interface.visualization.mission_renderer import MissionRenderer, MissionState

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"BOWL": 8.6, "PLATE": 9.8, "CHEESE": 11.0, "WINE": 12.2, "STOVE": 14.0, "CANCEL": 15.8}


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("step5d_demo", ROOT / "scripts" / "step5d_narrated_demo.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["step5d_demo"] = module
    spec.loader.exec_module(module)
    return module


def _alignment(text, seconds_per_char=0.05):
    return {"characters": list(text), "character_start_times_seconds": [i * seconds_per_char for i in range(len(text))],
            "character_end_times_seconds": [(i + 1) * seconds_per_char for i in range(len(text))]}


def test_word_starts_from_character_timings(demo):
    assert demo.word_starts(_alignment("Our decoder  listens.")) == pytest.approx([0.0, 0.2, 0.65])


def test_captions_light_up_spoken_words_and_colour_keywords(demo):
    line = demo.Line("x", ["put", "the", "bowl", "on", "the", "stove."], [0, 0.2, 0.4, 0.6, 0.8, 1.0], 1.3,
                     np.zeros(10, np.float32))
    caption = demo.caption_at(line, 0.45)
    assert [spoken for _, _, spoken in caption] == [True, True, True, False, False, False]
    assert caption[2][1] == demo.KEYWORDS["bowl"] and caption[5][1] == demo.KEYWORDS["stove"]  # punctuation stripped
    assert caption[0][1] is None


def test_word_cues_switch_the_spotlight_when_the_word_is_spoken(demo):
    words = "But this is a simulator. NVIDIA Cosmos keeps it, and repaints the world.".split()
    line = demo.Line("s10_cosmos", words, [0.3 * i for i in range(len(words))], 4.0, np.zeros(1, np.float32))
    assert demo.cue_focus(line, 0.0) == ("x1",)
    assert demo.cue_focus(line, 0.3 * words.index("Cosmos")) == ("x2",)
    assert demo.cue_focus(line, 3.9) == ("x3",)


def test_every_cue_word_exists_in_the_script(demo):
    import yaml

    texts = {s["id"]: s["text"].split() for s in yaml.safe_load((ROOT / "scripts/step5d_narration.yaml").read_text())["segments"]}
    for segment, cues in demo.CUES.items():
        for word, _ in cues:
            assert word in texts[segment], f"{segment}: cue word {word!r} not in the narration"


def test_beats_hold_until_the_line_finishes(demo):
    line = demo.Line("x", ["a"], [0.0], 2.0, np.zeros(1, np.float32))
    beat = demo.fit(demo.Beat([MissionState()] * 10, line, anchor=5))
    assert len(beat.frames) == 5 + int(np.ceil((2.0 + demo.TAIL) * demo.FPS))
    long = demo.fit(demo.Beat([MissionState()] * 500, line))
    assert len(long.frames) == 500  # never trimmed


def test_ms_copies_history_so_later_commands_do_not_leak_back(demo):
    history = ["#1 saw BOWL"]
    state = demo.ms({"history": history}, command=1)
    history.append("#5 saw STOVE")
    assert state.history == ["#1 saw BOWL"]


@pytest.mark.parametrize("scene, extra", [("idea", {}), ("kitchen", {"stimulus": True, "intended": "BOWL"}),
                                          ("explain", {"explain_frames": [np.zeros((8, 8, 3), np.uint8)] * 3})])
def test_new_scenes_render_with_spotlight_and_captions(scene, extra):
    renderer = MissionRenderer(TARGETS, ["O1", "Oz", "O2", "PO3", "PO4", "PO7", "PO8", "POz"], 0.075)
    caption = [(w, None, i < 3) for i, w in enumerate(("word " * 40).split())]
    focus = {"idea": ("i2",), "kitchen": ("stimulus",), "explain": ("x3",)}[scene]
    frame = renderer.frame(MissionState(scene=scene, focus=focus, caption=caption, clock=0.4, **extra))
    renderer.close()
    assert frame.shape == (1080, 1920, 3) and frame.std() > 1
