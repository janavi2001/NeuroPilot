"""Step 5 rented-GPU safety: watchdog decisions, remote waiting, on-instance self-destruct.

The end-to-end tests run the real scripts against a fake ``vastai`` on PATH, so
the destroy path is exercised without renting anything.
"""

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cloud.vast import (DESTROY_DEADLINE, DESTROY_LOW_CREDIT, DONE, FAILED, GONE, RUNNING, UNREACHABLE, WATCH,
                        RemoteWait, VastOutputError, classify_remote, guard_action, instance_present,
                        parse_credit, parse_instance_ids)

ROOT = Path(__file__).resolve().parents[1]
# Shape of real `vastai show instances --raw` output (fields observed 2026-11-15).
REAL_SHAPED = json.dumps([{"id": 60720037, "actual_status": "exited", "cur_state": "stopped",
                           "gpu_name": "RTX 4090", "disk_space": 120.0, "dph_total": 0.4711}], indent=4)


def test_parses_real_shaped_listing_and_empty_listing():
    assert parse_instance_ids(REAL_SHAPED) == {60720037}
    assert parse_instance_ids("[]") == set()
    assert instance_present(REAL_SHAPED, 60720037) is True
    assert instance_present("[]", 60720037) is False


def test_regression_compact_json_is_still_present():
    # 2026-11-14: a text grep for '"id": 60720037' missed the instance and the watchdog quit early.
    assert instance_present('[{"id":60720037,"cur_state":"running"}]', 60720037) is True


@pytest.mark.parametrize("text", [None, "", "Error: 429 Too Many Requests", "{}", '{"instances": 3}', '[{"x": 1}]'])
def test_unreadable_listing_is_unknown_never_gone(text):
    assert instance_present(text, 60720037) is None
    assert guard_action(0, 100, instance_present(text, 60720037), 5.0, 1.5) == WATCH
    if text is not None:
        with pytest.raises(VastOutputError):
            parse_instance_ids(text)


def test_credit_parsing():
    assert parse_credit('{"credit": 5.5937, "email": "x"}') == pytest.approx(5.5937)
    assert parse_credit("oops") is None and parse_credit(None) is None and parse_credit("{}") is None


@pytest.mark.parametrize("now, present, credit, expected", [
    (10, True, 5.0, WATCH),
    (10, None, None, WATCH),
    (10, False, 5.0, GONE),
    (100, True, 5.0, DESTROY_DEADLINE),
    (100, None, None, DESTROY_DEADLINE),  # unreadable at the deadline still destroys
    (10, True, 1.0, DESTROY_LOW_CREDIT),
    (10, None, 1.0, DESTROY_LOW_CREDIT),
])
def test_guard_action(now, present, credit, expected):
    assert guard_action(now, 100, present, credit, 1.5) == expected


@pytest.mark.parametrize("code, out, expected", [
    (0, "DONE\n", DONE), (0, "Welcome\nRUNNING\n", RUNNING), (0, "STOPPED", FAILED),
    (255, "", UNREACHABLE), (0, "", UNREACHABLE), (0, "bash: error", UNREACHABLE), (255, "RUNNING", UNREACHABLE),
])
def test_classify_remote(code, out, expected):
    assert classify_remote(code, out) == expected


def test_remote_wait_gives_up_on_repeated_unreachable_instead_of_hanging():
    # 2026-11-14: the setup watcher looped forever once the instance stopped answering.
    waiter = RemoteWait(3)
    assert [waiter.update(s) for s in (UNREACHABLE, UNREACHABLE, RUNNING, UNREACHABLE, UNREACHABLE)] == [None] * 5
    assert waiter.update(UNREACHABLE) == UNREACHABLE
    assert RemoteWait(5).update(DONE) == DONE and RemoteWait(5).update(FAILED) == FAILED


FAKE_VASTAI = r"""#!/usr/bin/env bash
# Fake vastai: state dir holds 'present' (instance exists), 'listing' (override output), calls log.
state="$FAKE_VASTAI_STATE"
echo "$*" >> "$state/calls"
case "$1 $2" in
  "show instances")
    if [ -f "$state/listing" ]; then cat "$state/listing"; exit 0; fi
    if [ -f "$state/present" ]; then echo '[{"id":60720037,"cur_state":"running"}]'; else echo '[]'; fi ;;
  "show user") cat "$state/credit" 2>/dev/null || echo '{"credit": 5.0}' ;;
  "show instance") [ "$4" = "--api-key" ] && [ "$5" = "good-key" ] || exit 1; echo '{"id":60720037}' ;;
  "destroy instance")
    rm -f "$state/present" "$state/listing"
    if [ -n "${FAKE_KILL_PARENT:-}" ]; then kill "$PPID"; fi ;;
esac
"""


@pytest.fixture
def fake_vastai(tmp_path):
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    bin_dir.mkdir()
    state.mkdir()
    script = bin_dir / "vastai"
    script.write_text(FAKE_VASTAI)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    (state / "present").touch()
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_VASTAI_STATE": str(state)}
    return script, state, env


def _guard(env, tmp_path, *extra):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "step5_vast_guard.py"), "--instance", "60720037",
                           "--interval", "0.1", "--log", str(tmp_path / "guard.log"), *extra],
                          env=env, capture_output=True, text=True, timeout=60)


def test_guard_check_mode_never_destroys(fake_vastai, tmp_path):
    _, state, env = fake_vastai
    result = _guard(env, tmp_path, "--check", "--hours", "0")
    assert "present=True credit=$5.00 action=destroy:deadline" in result.stdout
    assert (state / "present").exists()
    assert not any(line.startswith("destroy") for line in (state / "calls").read_text().splitlines())


def test_guard_destroys_at_deadline_even_when_listing_is_unreadable(fake_vastai, tmp_path):
    _, state, env = fake_vastai
    (state / "listing").write_text("Error: 502 Bad Gateway")  # the guard must not read this as gone
    result = _guard(env, tmp_path, "--hours", "0")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "destroy instance 60720037 -y" in (state / "calls").read_text()
    assert not (state / "present").exists()
    assert "confirmed destroyed" in (tmp_path / "guard.log").read_text()


def test_guard_destroys_when_credit_runs_low(fake_vastai, tmp_path):
    _, state, env = fake_vastai
    (state / "credit").write_text('{"credit": 1.20}')
    result = _guard(env, tmp_path, "--hours", "4", "--min-credit", "1.5")
    assert result.returncode == 0
    assert "destroy:low-credit" in (tmp_path / "guard.log").read_text()
    assert not (state / "present").exists()


def test_guard_exits_only_after_two_readable_listings_without_instance(fake_vastai, tmp_path):
    _, state, env = fake_vastai
    (state / "present").unlink()
    result = _guard(env, tmp_path, "--hours", "4")
    assert result.returncode == 0
    calls = (state / "calls").read_text()
    assert calls.count("show instances") == 2 and "destroy" not in calls


SELF_DESTRUCT = ROOT / "scripts" / "step5_self_destruct.sh"


def _environ(tmp_path, key):
    path = tmp_path / "environ"
    path.write_bytes(b"PATH=/usr/bin\0CONTAINER_ID=60720037\0CONTAINER_API_KEY=" + key.encode() + b"\0")
    return path


def test_self_destruct_reads_pid1_env_verifies_key_and_destroys(fake_vastai, tmp_path):
    script, state, env = fake_vastai
    env = {**env, "VASTAI": str(script), "SELF_DESTRUCT_ENVIRON": str(_environ(tmp_path, "good-key")),
           "SELF_DESTRUCT_LOG": str(tmp_path / "sd.log"), "FAKE_KILL_PARENT": "1"}
    result = subprocess.run(["bash", str(SELF_DESTRUCT), "1"], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0 and "SELF_DESTRUCT_ARMED instance=60720037" in result.stdout
    assert "good-key" not in result.stdout  # the key is never printed
    deadline = time.time() + 15
    while (state / "present").exists() and time.time() < deadline:
        time.sleep(0.2)
    assert not (state / "present").exists()
    assert "destroy instance 60720037 --api-key good-key -y" in (state / "calls").read_text()


def test_self_destruct_refuses_when_key_rejected_or_missing(fake_vastai, tmp_path):
    script, state, env = fake_vastai
    base = {**env, "VASTAI": str(script), "SELF_DESTRUCT_LOG": str(tmp_path / "sd.log")}
    rejected = subprocess.run(["bash", str(SELF_DESTRUCT), "1"], capture_output=True, text=True, timeout=20,
                              env={**base, "SELF_DESTRUCT_ENVIRON": str(_environ(tmp_path, "bad-key"))})
    assert rejected.returncode == 1 and "SELF_DESTRUCT_KEY_REJECTED" in rejected.stdout
    missing = subprocess.run(["bash", str(SELF_DESTRUCT), "1"], capture_output=True, text=True, timeout=20,
                             env={**base, "SELF_DESTRUCT_ENVIRON": str(tmp_path / "nope")})
    assert missing.returncode == 1 and "SELF_DESTRUCT_UNAVAILABLE" in missing.stdout
    time.sleep(1.5)
    assert (state / "present").exists()


def _probe(tmp_path, log_text, *, job_running):
    spec = __import__("importlib.util").util.spec_from_file_location(
        "step5_wait_remote", ROOT / "scripts" / "step5_wait_remote.py")
    module = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(module)
    log = tmp_path / "job.log"
    log.write_text(log_text)
    job = None
    if job_running:
        script = tmp_path / "np_fake_job.sh"
        script.write_text("sleep 30\n")
        job = subprocess.Popen(["bash", str(script)])
    try:
        # Run the probe the way ssh does: as a shell command whose own text names the process.
        out = subprocess.run(["bash", "-c", module.probe_command(str(log), "JOB_DONE", "np_fake_job.sh")],
                             capture_output=True, text=True, timeout=20)
        return classify_remote(out.returncode, out.stdout)
    finally:
        if job:
            job.kill()


def test_probe_reports_dead_job_as_failed_not_running(tmp_path):
    # 2026-11-19: pgrep -f matched the probe's own shell, so a crashed setup read as RUNNING for 55 min.
    assert _probe(tmp_path, "boom\n", job_running=False) == FAILED
    assert _probe(tmp_path, "working\n", job_running=True) == RUNNING
    assert _probe(tmp_path, "x\nJOB_DONE\n", job_running=False) == DONE
