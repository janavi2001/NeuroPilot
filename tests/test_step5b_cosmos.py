import importlib.util
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from cosmos.controls import (SIZE, build_control, canny_edges, edge_agreement, read_frames,  # noqa: E402
                             third_person_view, usable_frame_count)
from cosmos.scenarios import WORLDS, build_prompt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GR00T_VIDEO = sorted((ROOT / "artifacts/step5_gr00t/remote_results/bci_02_PLATE/videos").glob("*.mp4"))
# Top-level keys of NVIDIA's cosmos3-inference-transfer assets/edge/prompt.json.
NVIDIA_PROMPT_KEYS = {
    "subjects", "background_setting", "lighting", "aesthetics", "cinematography", "style_medium", "artistic_style",
    "context", "actions", "text_and_signage_elements", "segments", "transitions", "temporal_caption",
    "audio_description", "resolution", "aspect_ratio", "duration", "fps",
}


@pytest.mark.parametrize("n, expected", [(1, 1), (4, 1), (5, 5), (43, 41), (63, 61), (75, 73), (121, 121)])
def test_frame_count_is_4k_plus_1(n, expected):
    assert usable_frame_count(n) == expected
    with pytest.raises(ValueError):
        usable_frame_count(0)


def test_third_person_view_is_left_square_above_caption():
    frames = np.zeros((3, 292, 512, 3), np.uint8)
    frames[:, :256, :256] = 7
    view = third_person_view(frames)
    assert view.shape == (3, 256, 256, 3) and (view == 7).all()
    with pytest.raises(ValueError):
        third_person_view(np.zeros((1, 100, 512, 3), np.uint8))


def _square_video(offset=0):
    frames = np.zeros((5, 64, 64, 3), np.uint8)
    frames[:, 20 + offset:44 + offset, 20:44] = 255
    return frames


def test_canny_matches_nvidia_medium_preset_shape_and_values():
    edges = canny_edges(_square_video())
    assert edges.shape == (5, 64, 64, 3)
    assert set(np.unique(edges)) == {0, 255}
    assert (edges[..., 0] == edges[..., 2]).all()
    expected = cv2.Canny(cv2.cvtColor(_square_video()[0], cv2.COLOR_RGB2GRAY), 100, 200)
    assert (edges[0, ..., 0] == expected).all()


def test_edge_agreement_rewards_alignment_and_rejects_noise():
    control = canny_edges(_square_video())
    assert edge_agreement(control, _square_video())["f1"] == 1.0
    assert edge_agreement(control, _square_video(offset=2))["f1"] > 0.9  # within the 3 px tolerance
    noise = np.random.default_rng(0).integers(0, 256, (5, 64, 64, 3), dtype=np.uint8)
    assert edge_agreement(control, noise)["precision"] < 0.3
    with pytest.raises(ValueError):
        edge_agreement(control, _square_video()[:3])


def test_prompt_follows_nvidia_schema_for_every_world():
    for world in WORLDS.values():
        prompt = build_prompt("push the plate to the front of the stove", world, 73, 20.0)
        assert set(prompt) == NVIDIA_PROMPT_KEYS
        assert prompt["subjects"][0]["action"] == "push the plate to the front of the stove"
        assert prompt["background_setting"] == world.background
        assert prompt["duration"] == "3.6s" and prompt["aspect_ratio"] == "1,1"
    assert set(WORLDS) == {"modern_kitchen", "warehouse", "office", "night_kitchen"}


def test_jobs_put_every_episode_in_the_first_world_then_the_hero_everywhere():
    spec = importlib.util.spec_from_file_location("prep", ROOT / "scripts" / "step5b_cosmos_prepare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clips = ["a", "hero", "c"]
    worlds = ["kitchen", "w2", "w3"]
    jobs = module.ordered_jobs(clips, worlds, "hero")
    assert jobs[:3] == [("a", "kitchen"), ("hero", "kitchen"), ("c", "kitchen")]
    assert jobs[3:5] == [("hero", "w2"), ("hero", "w3")]
    assert len(jobs) == len(set(jobs)) == 9


@pytest.mark.skipif(not GR00T_VIDEO, reason="Step 5a GR00T videos not present")
def test_real_gr00t_episode_becomes_a_valid_control_video(tmp_path):
    meta = build_control(GR00T_VIDEO[0], tmp_path)
    edges, fps = read_frames(tmp_path / "control_edge.mp4")
    rgb, _ = read_frames(tmp_path / "rgb.mp4")
    assert meta["num_frames"] == 73 and fps == 20
    assert edges.shape == rgb.shape == (73, SIZE, SIZE, 3)
    assert 0.005 < meta["edge_pixel_fraction"] < 0.1
    assert edge_agreement((edges > 127).astype(np.uint8) * 255, rgb)["f1"] > 0.8  # control matches its own source
