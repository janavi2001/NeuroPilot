#!/usr/bin/env python3
"""Local watchdog for a rented Vast.ai instance: hard time cap and minimum-credit floor.

Every tick it reads the instance list and the account credit as JSON. When the
deadline passes or credit drops below the floor it copies results back, then
destroys the instance and keeps retrying until a readable listing shows it is
gone. It only exits early when two readable listings in a row lack the
instance; an unreadable listing never counts as gone.

  --check   one tick, print what it sees and would do, never destroys.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloud.vast import (DESTROY_DEADLINE, DESTROY_LOW_CREDIT, GONE, guard_action, instance_present,  # noqa: E402
                        parse_credit)


class Guard:
    def __init__(self, args) -> None:
        self.args = args
        self.vastai = shutil.which("vastai")
        if self.vastai is None:
            raise SystemExit("vastai CLI not found on PATH")
        self.log_path = Path(args.log)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        with self.log_path.open("a") as handle:
            handle.write(line + "\n")

    def _vastai(self, *command: str, timeout: float = 60) -> str | None:
        try:
            completed = subprocess.run([self.vastai, *command], capture_output=True, text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, OSError) as error:
            self.log(f"vastai {command[:2]} failed: {error}")
            return None
        if completed.returncode != 0:
            self.log(f"vastai {command[:2]} exit {completed.returncode}: {completed.stderr.strip()[:200]}")
            return None
        return completed.stdout

    def present(self) -> bool | None:
        return instance_present(self._vastai("show", "instances", "--raw"), self.args.instance)

    def credit(self) -> float | None:
        return parse_credit(self._vastai("show", "user", "--raw"))

    def copy_results(self) -> None:
        if not self.args.ssh_host:
            return
        destination = Path(self.args.results_dir)
        destination.mkdir(parents=True, exist_ok=True)
        ssh = f"ssh -p {self.args.ssh_port} -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
        command = ["rsync", "-az", "-e", ssh, f"{self.args.ssh_host}:{self.args.remote_results}/", f"{destination}/"]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
            self.log(f"copy results exit {completed.returncode} {completed.stderr.strip()[:200]}")
        except subprocess.TimeoutExpired:
            self.log("copy results timed out after 900 s; destroying anyway")

    def destroy_until_gone(self, attempts: int = 20) -> bool:
        for attempt in range(1, attempts + 1):
            self._vastai("destroy", "instance", str(self.args.instance), "-y")
            time.sleep(min(10, self.args.interval))
            if self.present() is False:
                self.log(f"instance {self.args.instance} confirmed destroyed (attempt {attempt})")
                return True
            self.log(f"instance {self.args.instance} not confirmed gone after attempt {attempt}")
            time.sleep(min(30, self.args.interval))
        self.log(f"FAILED to confirm destruction of {self.args.instance}; destroy it manually")
        return False

    def run(self) -> int:
        args = self.args
        deadline = time.time() + args.hours * 3600
        self.log(f"guard armed: instance {args.instance}, deadline {datetime.fromtimestamp(deadline):%Y-%m-%d %H:%M}, "
                 f"min credit ${args.min_credit:.2f}")
        gone_in_a_row = 0
        while True:
            present, credit = self.present(), self.credit()
            action = guard_action(time.time(), deadline, present, credit, args.min_credit)
            credit_text = "unknown" if credit is None else f"${credit:.2f}"
            if args.check:
                minutes = (deadline - time.time()) / 60
                print(f"present={present} credit={credit_text} action={action} (deadline in {minutes:.0f} min)")
                return 0
            if action == GONE:
                gone_in_a_row += 1
                if gone_in_a_row >= 2:
                    self.log(f"instance {args.instance} no longer listed (two readable listings); guard exiting")
                    return 0
            else:
                gone_in_a_row = 0
            if action in (DESTROY_DEADLINE, DESTROY_LOW_CREDIT):
                self.log(f"{action}: credit {credit_text}; copying results then destroying {args.instance}")
                self.copy_results()
                return 0 if self.destroy_until_gone() else 1
            if present is None or credit is None:
                self.log(f"tick: present={present} credit={credit_text} (unreadable value is not treated as gone)")
            time.sleep(args.interval)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance", type=int, required=True)
    parser.add_argument("--hours", type=float, default=4.0, help="hard cap from now")
    parser.add_argument("--min-credit", type=float, default=1.5, help="destroy when credit falls below this (USD)")
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--ssh-host", default="", help="e.g. root@1.2.3.4, used to copy results before destroying")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--remote-results", default="/root/neuropilot_results")
    parser.add_argument("--results-dir", default="artifacts/step5_gr00t/remote_results")
    parser.add_argument("--log", default="artifacts/step5_gr00t/guard.log")
    parser.add_argument("--check", action="store_true")
    return Guard(parser.parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
