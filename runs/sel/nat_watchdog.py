"""Independent native deadline cleanup. Children are separate setsid groups."""
import argparse
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from nat_runtime import deadline


def remove_owned(owner):
    if not re.fullmatch(r'rtm-nat-[a-zA-Z0-9-]+', owner):
        raise ValueError('invalid native container owner')
    try:
        result = subprocess.run(['docker', 'ps', '-a', '--format', '{{.Names}}'],
                                capture_output=True, text=True, timeout=5)
        names = [name for name in result.stdout.splitlines() if name.startswith(owner + '-')]
        if names:
            # One bounded call for all owned containers, so a stalled Docker
            # command cannot add a separate timeout per container before KILL.
            subprocess.run(['docker', 'rm', '-f', *names], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass


def start_ticks(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def signal_children(pid_file, sig):
    try:
        lines = Path(pid_file).read_text().splitlines()
    except FileNotFoundError:
        lines = []
    for line in lines:
        fields = line.split()
        if len(fields) != 2:
            continue
        pid, started = int(fields[0]), fields[1]
        if pid <= 1 or pid in (os.getpid(), os.getpgrp()):
            continue
        current = start_ticks(pid)
        if current is not None and current != started:
            continue  # never signal a reused PID
        # The leader can have exited while grandchildren still own its group.
        for send in (os.killpg, os.kill):
            try:
                send(pid, sig)
            except ProcessLookupError:
                pass


def enforce(owner, pid_file, parent=None, parent_ticks=None, grace=20):
    # Independent of Bash traps/waits. Stop GPU owners directly on deadline.
    if not re.fullmatch(r'rtm-nat-[a-zA-Z0-9-]+', owner):
        raise ValueError('invalid native container owner')
    started = time.monotonic()
    signal_children(pid_file, signal.SIGTERM)
    remove_owned(owner)
    if parent is not None and parent_ticks is not None and start_ticks(parent) == parent_ticks:
        try:
            os.kill(parent, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(max(0, grace - (time.monotonic() - started)))
    signal_children(pid_file, signal.SIGKILL)
    remove_owned(owner)  # catches a docker run racing the first sweep
    if parent is not None and parent_ticks is not None and start_ticks(parent) == parent_ticks:
        try:
            os.kill(parent, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--owner', required=True)
    ap.add_argument('--pid-file', required=True)
    ap.add_argument('--parent', required=True, type=int)
    ap.add_argument('--deadline', type=float, default=None)
    ap.add_argument('--cleanup', action='store_true')
    a = ap.parse_args()
    marker = Path(a.pid_file + '.expired')
    try:
        if a.cleanup:
            enforce(a.owner, a.pid_file)
        else:
            end = deadline() if a.deadline is None else a.deadline
            parent_ticks = start_ticks(a.parent)
            while parent_ticks is not None and time.time() < end and start_ticks(a.parent) == parent_ticks:
                time.sleep(min(1, max(0, end - time.time())))
            marker.touch()
            print('HARD STOP: native deadline/parent exit; removing owned containers and children', flush=True)
            enforce(a.owner, a.pid_file, a.parent, parent_ticks)
    finally:
        Path(a.pid_file).unlink(missing_ok=True)
        marker.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
