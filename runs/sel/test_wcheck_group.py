"""Round/gold alignment and byte regression against wcheck.py.bak-1108 (CPU)."""
import ast
import collections
import json
import os
from pathlib import Path
import subprocess
import sys

from k1 import rounds_of

HERE = Path(__file__).resolve().parent


def rows_function():
    # wcheck is a CLI with import-time fitting. Extract the real function without
    # importing torch or running unrelated reports in this focused alignment test.
    tree = ast.parse((HERE / "wcheck.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "rows")
    scope = dict(json=json, collections=collections)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "wcheck.py", "exec"), scope)
    return scope["rows"]


def test_consecutive_speakers_sessions_and_gold_union(tmp_path):
    sessions = [[], [dict(role="assistant", content="orphan assistant"),
                     dict(role="user", content="[Group 1] A: first"),
                     dict(role="user", content="[Group 1] B: second", kind="repeat"),
                     dict(role="user", content="[Group 1] C: third"),
                     dict(role="assistant", content="paired answer"),
                     dict(role="assistant", content="standalone answer"),
                     dict(role="user", content="[Group 1] A: fourth", kind="fu"),
                     dict(role="user", content="last")],
                [dict(role="assistant", content="new session orphan"),
                 dict(role="user", content="final user")]]
    q = dict(question_id="w1_a", haystack_sessions=sessions, haystack_dates=["d0", "d1", "d2"])
    R = rounds_of(q)
    assert len(R) == 9
    qs = [q, dict(q, question_id="w1_b")]
    (tmp_path / "q.json").write_text(json.dumps(qs))
    for q_, positives in zip(qs, ([1, 4], [3, 8])):
        (tmp_path / (q_["question_id"] + ".json")).write_text(json.dumps(dict(gold=[i in positives for i in range(len(R))])))
    result = rows_function()(tmp_path / "q.json", tmp_path, lambda q: q["question_id"].split("_")[0])
    # repeat/fu are excluded only AFTER index assignment; assistant-only rounds
    # remain separate, as do user messages on opposite sides of a session edge.
    assert [r[1] for r in result] == ["orphan assistant", "[Group 1] A: first", "[Group 1] C: third", "standalone answer", "last", "new session orphan", "final user"]
    assert [r[4] for r in result] == [0, 1, 1, 1, 0, 0, 1]
    assert result[2][2] == "paired answer"
    assert [r[3] for r in result] == [i / 9 for i in (0, 1, 3, 4, 6, 7, 8)]


def test_one_to_one_full_stdout_byte_regression(tmp_path):
    qs = []
    for wid in range(3):
        turns = []
        for i in range(9):
            turns += [dict(role="user", content=f"hey project {wid} update {i}? " + "text " * (i + 1), kind="repeat" if i == 7 else "ev"),
                      dict(role="assistant", content="ack " * (i + 1))]
        qs.append(dict(question_id=f"w{wid}_q", haystack_sessions=[turns[:8], turns[8:]], haystack_dates=["d1", "d2"]))
    for name in ("beam100k_g.json", "env_legacy.json"):
        (tmp_path / name).write_text(json.dumps(qs))
    for dirname in ("pool_c", "pool_legacy"):
        (tmp_path / dirname).mkdir()
        for q in qs:
            (tmp_path / dirname / (q["question_id"] + ".json")).write_text(json.dumps(dict(gold=[i in (1, 4, 8) for i in range(9)])))
    (tmp_path / "datacheck.py").symlink_to(HERE / "datacheck.py")
    # Prevent torch import and GPU detection altogether, even on a GPU machine.
    harness = "import runpy,sys,types; sys.modules['torch']=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:False)); sys.argv=[sys.argv[1],'legacy']; runpy.run_path(sys.argv[0],run_name='__main__')"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    results = [subprocess.run([sys.executable, "-c", harness, str(HERE / script)], cwd=tmp_path, env=env, capture_output=True)
               for script in ("wcheck.py.bak-1108", "wcheck.py")]
    assert results[0].stdout  # A real report, not two empty failures.
    assert results[0].returncode == results[1].returncode
    assert results[0].stdout == results[1].stdout
    assert results[0].stderr == results[1].stderr == b""
