"""Mock Docker only: process-group/deferred-trap behavior uses real OS processes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

import nat_watchdog as wd

HERE = Path(__file__).resolve().parent


def docker_mock(tmp_path, monkeypatch):
    binary = tmp_path / 'docker'
    binary.write_text('#!/bin/sh\n'
                      'if [ "$1" = ps ]; then\n'
                      '  printf "%s\\n" rtm-nat-test-emb rtm-nat-test-train rtm-nat-test2-emb rtm-nat-other-emb rtm-v3-q4 rtm-nat-test\n'
                      'else\n'
                      '  echo "$*" >> "$DOCKER_LOG"\n'
                      'fi\n')
    binary.chmod(0o755)
    log = tmp_path / 'docker.log'
    monkeypatch.setenv('DOCKER_LOG', str(log))
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ['PATH'])
    return log


def test_strict_owner_prefix_and_invalid_owner(tmp_path, monkeypatch):
    log = docker_mock(tmp_path, monkeypatch)
    wd.remove_owned('rtm-nat-test')
    assert log.read_text().splitlines() == ['rm -f rtm-nat-test-emb rtm-nat-test-train']
    for invalid in ('', 'rtm-v3', 'rtm-nat-*', 'rtm-nat-test;bad'):
        with pytest.raises(ValueError, match='owner'):
            wd.remove_owned(invalid)


def test_term_kill_20s_both_pid_and_group_and_pid_reuse(tmp_path, monkeypatch):
    pids = tmp_path / 'pids'
    pids.write_text('12345 original\n54321 old\n')
    events = []
    monkeypatch.setattr(wd, 'start_ticks', lambda p: 'original' if p == 12345 else 'new')
    monkeypatch.setattr(wd.os, 'killpg', lambda p, s: events.append(('group', p, s)))
    monkeypatch.setattr(wd.os, 'kill', lambda p, s: events.append(('pid', p, s)))
    monkeypatch.setattr(wd, 'remove_owned', lambda owner: events.append(('docker', owner)))
    monkeypatch.setattr(wd.time, 'monotonic', lambda: 10)
    monkeypatch.setattr(wd.time, 'sleep', lambda t: events.append(('sleep', t)))
    wd.enforce('rtm-nat-test', pids)
    assert ('sleep', 20) in events
    for sig in (signal.SIGTERM, signal.SIGKILL):
        assert ('group', 12345, sig) in events and ('pid', 12345, sig) in events
    assert all(len(e) < 3 or e[1] != 54321 for e in events)
    assert events.index(('group', 12345, signal.SIGTERM)) < events.index(('sleep', 20))
    assert events.index(('group', 12345, signal.SIGKILL)) > events.index(('sleep', 20))
    assert events.count(('docker', 'rtm-nat-test')) == 2


def wait_for(predicate, seconds=5):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError('timed out waiting for process fixture')


def dead(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[0] == 'Z'
    except FileNotFoundError:
        return True


def test_deadline_removes_docker_despite_deferred_trap_and_kills_descendants(tmp_path, monkeypatch):
    log = docker_mock(tmp_path, monkeypatch)
    # Bash defers TERM while a foreground child ignores TERM. The child's own
    # descendant also ignores TERM; both must be killed as a group after 20s.
    stubborn = tmp_path / 'stubborn.py'
    stubborn.write_text('import os, signal, subprocess, sys, time\n'
                        'from pathlib import Path\n'
                        'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
                        'kid = subprocess.Popen([sys.executable, "-c", '
                        '"import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])\n'
                        'Path(sys.argv[1]).write_text(str(os.getpid()) + " " + str(kid.pid))\n'
                        'time.sleep(60)\n')
    launch = tmp_path / 'parent.sh'
    launch.write_text('trap \'echo deferred > "$1/trap"\' TERM\n'
                      'setsid "$2" "$1/stubborn.py" "$1/children"\n')
    parent = subprocess.Popen(['bash', str(launch), str(tmp_path), sys.executable],
                              start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    watcher = None
    leader = grandchild = None
    try:
        wait_for(lambda: (tmp_path / 'children').exists())
        leader, grandchild = map(int, (tmp_path / 'children').read_text().split())
        pids = tmp_path / 'pids'
        pids.write_text(f'{leader} {wd.start_ticks(leader)}\n')
        watcher = subprocess.Popen([sys.executable, '-B', str(HERE / 'nat_watchdog.py'),
                                    '--owner', 'rtm-nat-test', '--pid-file', str(pids),
                                    '--parent', str(parent.pid), '--deadline', str(time.time() + .15)],
                                   start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        wait_for(lambda: log.exists() and 'rm -f rtm-nat-test-emb' in log.read_text())
        assert not (tmp_path / 'trap').exists()  # independent of parent's trap
        assert not dead(leader) and not dead(grandchild)
        watcher.communicate(timeout=26)
        assert watcher.returncode == 0
        parent.wait(timeout=3)
        wait_for(lambda: dead(leader) and dead(grandchild))
        assert log.read_text().splitlines() == ['rm -f rtm-nat-test-emb rtm-nat-test-train'] * 2
    finally:
        for pid in (leader, parent.pid):
            if pid:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if watcher and watcher.poll() is None:
            watcher.kill()
            watcher.wait()
        parent.wait()


@pytest.mark.parametrize('mode', ('rebuild', 'train', 'trainj'))
def test_launcher_modes_skip_generation_and_pool(tmp_path, mode):
    # Exercise real Bash branching/tracking; all external actions are mocks.
    (tmp_path / 'run_nat_quick.sh').write_bytes((HERE / 'run_nat_quick.sh').read_bytes())
    (tmp_path / 's23.py').write_bytes((HERE / 's23.py').read_bytes())
    (tmp_path / 'provenance.json').write_text('{}')
    (tmp_path / 'nat_gpus47.ok').touch()
    binary = tmp_path / 'bin'
    binary.mkdir()
    python = binary / 'python3'
    python.write_text('#!' + sys.executable + '\n'
                      'import json, os, sys, time\n'
                      'a = sys.argv[1:]\n'
                      'with open("commands.jsonl", "a") as f: f.write(json.dumps(a) + "\\n")\n'
                      'if "nat_watchdog.py" in a:\n'
                      '    if "--cleanup" not in a: time.sleep(60)\n'
                      'elif "remaining" in a: print(999999)\n')
    python.chmod(0o755)
    docker = binary / 'docker'
    docker.write_text('#!/bin/sh\nexit 0\n')
    docker.chmod(0o755)
    env = {**os.environ, 'PATH': str(binary) + os.pathsep + os.environ['PATH'],
           'S23_STAGE1_PROVENANCE': str(tmp_path / 'provenance.json'), 'NAT_TOKENIZER': '/mock/tokenizer'}
    result = subprocess.run(['bash', str(tmp_path / 'run_nat_quick.sh'), mode], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [json.loads(s) for s in (tmp_path / 'commands.jsonl').read_text().splitlines()]
    assert not any('pool.py' in c for c in commands)
    converts = [c for c in commands if 'mine3_to_env.py' in c]
    if mode == 'rebuild':
        assert len(converts) == 1 and '--refilter' in converts[0]
        mk = next(c for c in commands if 'mkdata.py' in c)
        assert mk[2:5] == ['env_nat2.json', 'pool_nat', 'tr_nat.rebuild.jsonl']
        assert mk[-2:] == ['--ids', 'nat_tr2.txt']
        assert not any('s1' in c for c in commands)
    else:
        assert not converts and not any('mkdata.py' in c for c in commands)
        assert any('s1' in c for c in commands) == (mode == 'train')
        if mode == 'trainj':
            jobs = [c for c in commands if 's23nat.py' in c]
            assert len(jobs) == 6  # two dry-runs, two smoke, two full
            assert all(c[c.index('--arm') + 1] == 'natj' for c in jobs)
            for fold, gpus in (('loc', '0,1,2,3'), ('ev', '4,5,6,7')):
                training = [c for c in jobs if c[c.index('--fold') + 1] == fold and '--dry-run' not in c]
                assert len(training) == 2
                assert all(c[c.index('--gpus') + 1] == gpus for c in training)
                assert training[0][-2:] == ['--smoke-steps', '50']
                assert '--smoke-steps' not in training[1]
            checks = [c for c in commands if 'nat_runtime.py' in c and 'check' in c]
            assert all(c[c.index('--arm') + 1] == 'natj' for c in checks)
            assert not any('s1' in c for c in checks)
            summary = next(c for c in commands if 'nat_summary.py' in c)
            assert summary[-2:] == ['--arm', 'natj']
    assert sum('--dry-run' in c for c in commands) == 2
    assert any('nat_artifacts.py' in c and 'check' in c for c in commands)
