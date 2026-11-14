import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

from bci.nlb_data import DEFAULT_NWB, SIX_DIRECTION_TARGETS, direction_class
from bci.ssvep_stopping import StoppingRule
from interface.ssvep_selector import load_ssvep_config
from simulation.franka_environment import PANDA_XML

ROOT = Path(__file__).resolve().parents[1]
SIX = ROOT / "interface" / "ssvep_targets_6.yaml"
requires_franka = pytest.mark.skipif(not PANDA_XML.is_file(), reason="Franka model not fetched")
requires_nlb = pytest.mark.skipif(not DEFAULT_NWB.is_file(), reason="NLB MC_Maze_Large not downloaded")


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_six_target_config_is_complete_and_harmonic_safe():
    config = load_ssvep_config(SIX)
    assert list(config.targets) == ["BANANA", "APPLE", "CUP", "BOTTLE", "PHONE", "CANCEL"]
    assert config.objects == ("BANANA", "APPLE", "CUP", "BOTTLE", "PHONE")
    grid = {round(8.0 + 0.2 * i, 1) for i in range(40)}
    assert set(config.targets.values()) <= grid  # real BETA stimulus frequencies
    select = _script("step4f_select_frequencies")
    gaps = [select.harmonic_gap(a, b) for a, b in itertools.combinations(config.targets.values(), 2)]
    assert min(gaps) >= 1.2 - 1e-9
    assert config.margin_threshold == pytest.approx(0.075)
    assert StoppingRule.from_yaml(SIX) == StoppingRule(0.3, 0.17, 0.5)


def test_frequency_search_respects_the_gap():
    select = _script("step4f_select_frequencies")
    sets = select.candidate_sets(6, 1.2)
    assert len(sets) == 56
    assert not select.candidate_sets(6, 1.4)
    for frequencies in sets[:10]:
        assert min(select.harmonic_gap(a, b) for a, b in itertools.combinations(frequencies, 2)) >= 1.2 - 1e-9


def test_four_target_config_is_unchanged():
    config = load_ssvep_config()
    assert config.targets == {"APPLE": 9.8, "CUP": 11.8, "PHONE": 13.8, "CANCEL": 15.8}


@requires_franka
def test_three_object_scene_is_still_the_default():
    from simulation.franka_environment import FrankaSimulation

    simulation = FrankaSimulation()
    assert simulation.OBJECTS == ("APPLE", "CUP", "PHONE")
    assert simulation.model.opt.impratio == 1  # verified Step 4c physics untouched
    with pytest.raises(ValueError, match="Unknown scene"):
        FrankaSimulation(scene="kitchen")


@requires_franka
@pytest.mark.parametrize("target", ["BANANA", "APPLE", "CUP", "BOTTLE", "PHONE"])
def test_six_scene_physical_pick_of_every_object(target):
    from robot.franka_controller import FrankaPickController
    from simulation.franka_environment import FrankaSimulation

    simulation = FrankaSimulation(scene="six")
    result = FrankaPickController(simulation).execute("pick", target)
    assert result["grasp_success"] and result["fingers_in_contact_after_lift"]
    assert result["lift_delta"] > 0.15
    assert result["object_tcp_distance"] < 0.04  # measured at the grasped point (bottle neck)
    assert result["max_other_object_shift"] < 0.01
    assert result["lifted"]


@requires_franka
def test_bottle_does_not_creep_out_of_the_grip():
    from robot.franka_controller import FrankaPickController
    from simulation.franka_environment import FrankaSimulation

    simulation = FrankaSimulation(scene="six")
    heights = []
    controller = FrankaPickController(simulation, lambda phase: heights.append(simulation.position("BOTTLE")[2])
                                      if phase == "hold" else None)
    controller.execute("pick", "BOTTLE")
    assert abs(heights[-1] - heights[0]) < 0.005


@pytest.mark.parametrize("xy, expected", [
    ((-130, -13), "BANANA"), ((-126, 15), "APPLE"), ((-105, 76), "CUP"), ((12, 84), "BOTTLE"),
    ((103, 83), "PHONE"), ((123, 70), "PHONE"), ((123, -81), "CANCEL"),
])
def test_six_direction_mapping(xy, expected):
    assert direction_class(np.array(xy, dtype=float), "six") == expected
    with pytest.raises(ValueError):
        direction_class(np.array(xy, dtype=float), "eight")


@requires_nlb
def test_six_direction_decoder_on_held_out_trials():
    from bci.intracortical_decoder import DecoderWindow, ReachIntentDecoder
    from bci.nlb_data import load_mc_maze

    trials = load_mc_maze(scheme="six")
    assert set(trials.labels) == set(SIX_DIRECTION_TARGETS)
    train, val = np.flatnonzero(trials.split == "train"), np.flatnonzero(trials.split == "val")
    decoder = ReachIntentDecoder(SIX_DIRECTION_TARGETS, DecoderWindow(-0.4, 0.0, 0.1)).fit(trials, train)
    accuracy = np.mean([decoder.predict_trial(trials, i)["intent"] == trials.labels[i] for i in val])
    assert accuracy >= 0.85


def test_hero_stats_follow_the_six_target_results():
    import yaml

    hero = _script("step4c_hero_demo")
    results = yaml.safe_load(SIX.read_text())["results"]
    if not Path(results["eval"]).is_file():
        pytest.skip("run the Step 4f evaluations first")
    stats = hero.load_stats(results, early=False)
    assert stats["single-look accuracy"] == "87.1%"
    assert stats["wrong actions, two-look confirm"].startswith("0.8%")
