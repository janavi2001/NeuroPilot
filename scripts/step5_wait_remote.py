#!/usr/bin/env python3
"""Wait for a remote background job by its log marker; fail loudly instead of hanging.

Exit codes: 0 done, 1 job stopped without its marker, 2 host unreachable for
--max-unreachable probes in a row, 3 timed out. The log tail is printed on exit.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloud.vast import DONE, FAILED, UNREACHABLE, RemoteWait, classify_remote  # noqa: E402

EXIT_CODES = {DONE: 0, FAILED: 1, UNREACHABLE: 2}


def probe_command(log: str, marker: str, process: str) -> str:
    # "[s]etup.sh" matches "setup.sh" but not itself, so pgrep cannot match the
    # shell running this probe (2026-11-19: it did, and a dead job read as RUNNING).
    process = f"[{process[0]}]{process[1:]}"
    log, marker, process = shlex.quote(log), shlex.quote("^" + marker), shlex.quote(process)
    return (f"if grep -q {marker} {log}; then echo DONE; "
            f"elif pgrep -f {process} >/dev/null; then echo RUNNING; else echo STOPPED; fi")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--log", required=True, help="remote log file")
    parser.add_argument("--marker", required=True, help="line prefix written when the job succeeds")
    parser.add_argument("--process", required=True, help="pgrep -f pattern of the running job")
    parser.add_argument("--timeout-minutes", type=float, default=60)
    parser.add_argument("--max-unreachable", type=int, default=5)
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()

    ssh = ["ssh", "-p", str(args.ssh_port), "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", args.ssh_host]
    waiter = RemoteWait(args.max_unreachable)
    deadline = time.time() + args.timeout_minutes * 60
    outcome = None
    while outcome is None:
        try:
            completed = subprocess.run([*ssh, probe_command(args.log, args.marker, args.process)],
                                       capture_output=True, text=True, timeout=60)
            state = classify_remote(completed.returncode, completed.stdout)
        except subprocess.TimeoutExpired:
            state = UNREACHABLE
        print(f"{time.strftime('%H:%M:%S')} {state}", flush=True)
        outcome = waiter.update(state)
        if outcome is None and time.time() > deadline:
            print(f"TIMEOUT after {args.timeout_minutes} min")
            outcome = "TIMEOUT"
        if outcome is None:
            time.sleep(args.interval)
    print(f"final: {outcome}")
    tail = subprocess.run([*ssh, f"tail -30 {shlex.quote(args.log)}"], capture_output=True, text=True, timeout=60)
    print(tail.stdout or tail.stderr)
    return EXIT_CODES.get(outcome, 3)


if __name__ == "__main__":
    raise SystemExit(main())
