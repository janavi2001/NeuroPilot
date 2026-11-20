import importlib.util
from pathlib import Path

import pytest

from interface.ssvep_confirm import ConfirmingSelector
from interface.ssvep_selector import load_ssvep_config

ROOT = Path(__file__).resolve().parents[1]
KITCHEN = ROOT / "interface" / "ssvep_targets_kitchen.yaml"
SIX = ROOT / "interface" / "ssvep_targets_6.yaml"
# Official LIBERO-Goal task names (LIBERO libero_suite_task_map.py).
LIBERO_GOAL = {
    "open_the_middle_drawer_of_the_cabinet", "put_the_bowl_on_the_stove", "put_the_wine_bottle_on_top_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside", "put_the_bowl_on_top_of_the_cabinet",
    "push_the_plate_to_the_front_of_the_stove", "put_the_cream_cheese_in_the_bowl", "turn_on_the_stove",
    "put_the_bowl_on_the_plate", "put_the_wine_bottle_on_the_rack",
}


def _jobs():
    spec = importlib.util.spec_from_file_location("step5_gr00t_jobs", ROOT / "scripts" / "step5_gr00t_jobs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kitchen_config_reuses_the_calibrated_six_target_decoder():
    kitchen, six = load_ssvep_config(KITCHEN), load_ssvep_config(SIX)
    assert list(kitchen.targets.values()) == list(six.targets.values())
    assert (kitchen.channels, kitchen.window_seconds, kitchen.margin_threshold) == \
        (six.channels, six.window_seconds, six.margin_threshold)
    assert kitchen.objects == ("BOWL", "PLATE", "CHEESE", "WINE", "STOVE")


def test_every_kitchen_object_maps_to_an_official_libero_goal_task():
    jobs = _jobs()
    tasks = jobs.load_tasks(KITCHEN)
    assert set(tasks) == set(load_ssvep_config(KITCHEN).objects)
    names = {jobs.libero_env_name(text).removeprefix("libero_sim/") for text in tasks.values()}
    assert names <= LIBERO_GOAL and len(names) == 5


def test_baseline_jobs():
    jobs = _jobs()
    baseline = jobs.baseline_jobs(jobs.load_tasks(KITCHEN), episodes=10, seed=0)
    assert [j["label"] for j in baseline] == [f"baseline_{n}" for n in ("BOWL", "PLATE", "CHEESE", "WINE", "STOVE")]
    assert all(j["episodes"] == 10 and j["env_name"].startswith("libero_sim/") for j in baseline)


def _feed(*intents):
    queue = [{"intent": i} for i in intents]
    return lambda intended: queue.pop(0) if queue else None


def test_confirm_command_needs_two_agreeing_looks():
    jobs, selector = _jobs(), ConfirmingSelector(("BOWL", "PLATE"))
    looks, confirmed = jobs.confirm_command("BOWL", _feed("UNSURE", "BOWL", "BOWL"), selector, 4)
    assert confirmed == "BOWL" and [l["event"] for l in looks] == ["unsure", "candidate", "confirm"]
    _, confirmed = jobs.confirm_command("BOWL", _feed("PLATE", "PLATE"), selector, 4)
    assert confirmed == "PLATE"  # a wrong confirmation is returned, not hidden
    _, confirmed = jobs.confirm_command("BOWL", _feed("BOWL", "CANCEL", "PLATE", "BOWL"), selector, 4)
    assert confirmed is None  # max looks reached: robot stays idle
    _, confirmed = jobs.confirm_command("BOWL", _feed("BOWL"), selector, 4)
    assert confirmed is None  # trials ran out


def _beta_cached() -> bool:
    try:
        import mne
        import moabb  # noqa: F401
    except ImportError:
        return False
    root = Path(mne.get_config("MNE_DATA") or Path.home() / "mne_data")
    return (root / "MNE-liu2020beta-data" / "S1.mat").is_file()


@pytest.mark.skipif(not _beta_cached(), reason="real BETA S1 data is not downloaded")
def test_real_beta_replay_produces_libero_jobs():
    jobs = _jobs()
    decisions, created = jobs.bci_jobs(1, KITCHEN, ["BOWL", "PLATE", "CHEESE", "WINE", "STOVE"], 4, 100)
    assert len(decisions) == 5
    for decision in decisions:
        assert all(look["ground_truth"] == decision["intended"] for look in decision["looks"])
        assert 0 < decision["brain_seconds"] <= 4 * 2.0
    assert len(created) == sum(d["confirmed"] is not None for d in decisions)
    assert all(j["env_name"].removeprefix("libero_sim/") in LIBERO_GOAL for j in created)
