"""CPU-only speaker-negative safety and byte regression against the frozen backup."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from audit_groups import doc_user
from mkdata import speakers_of

HERE = Path(__file__).resolve().parent


@pytest.fixture
def data(tmp_path):
    pooldir = tmp_path / "pool"
    pooldir.mkdir()
    sessions = []
    stages = []
    for i in range(100):
        sp = "Bob" if i < 3 else ("Alice", "Bob", "Chen")[i % 3]
        stages.append("gA#0" if i == 0 else "gA#1" if i == 1 else "gB#2" if i == 2 else "support" if i == 3 else f"f{i}")
        sessions.append(dict(role="user", content=f"[Group 1] {sp}: round-{i}: information", stage=stages[-1], kind="repeat" if i == 4 else "fu" if i == 5 else "ev"))
    q = dict(question_id="w1_q", question_type="aggregation", question="I (Alice) need the project facts.", asker="Alice",
             question_date="2026-11-01", gold_stages=stages[:3], support_stages=["support"],
             haystack_sessions=[sessions[:41], sessions[41:]], haystack_dates=["d1", "d2"])
    pool = dict(stage=stages, gold=[i < 3 for i in range(100)], fused=list(range(95)), rr8={str(i): (100 - i) / 101 for i in range(70)})
    qfile = tmp_path / "q.json"
    qfile.write_text(json.dumps([q]))
    (pooldir / "w1_q.json").write_text(json.dumps(pool))
    (tmp_path / "fn.json").write_text(json.dumps({"w1_q": dict(support=[68, 69], ignore=[70])}))
    (tmp_path / "cands.json").write_text(json.dumps({"w1_q": {"8b": list(range(95)), "rand": list(range(68, 95))}}))
    return tmp_path, qfile, pooldir, q, pool


def run(data, script="mkdata.py", flags=(), name="out"):
    root, qfile, pooldir, _, _ = data
    out = root / (name + ".jsonl")
    env = dict(os.environ, PYTHONHASHSEED="0", CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1")
    env.pop("FMT_INST", None); env.pop("MASK_DATE", None)
    result = subprocess.run([sys.executable, str(HERE / script), str(qfile), str(pooldir), str(out), *map(str, flags)],
                            cwd=HERE, env=env, capture_output=True)
    return result, out


@pytest.mark.parametrize("flags", [
    ["--no-set", "--steps", "4"],
    ["--no-set", "--steps", "2", "--spk-neg", "0", "--K", "4"],
    ["--steps", "12", "--p-mixed", "1", "--seed", "7", "--ignore-kinds", "repeat,fu"],
    ["--steps", "30", "--kmax", "3", "--p-mixed", "0", "--seed", "19"],
    ["--no-set", "--fact-stage", "--hard-skip", "10", "--ignore-kinds", "repeat,fu", "--boost", "3", "--steps", "2"],
    ["--no-set", "--gold-rounds", "--steps", "2"],
    ["--teacher-only", "--n-groups", "3"],
])
def test_default_off_byte_regression(data, flags):
    old_flags = list(flags)
    if "--spk-neg" in old_flags:
        i = old_flags.index("--spk-neg")
        del old_flags[i:i + 2]  # the frozen script predates this opt-in flag
    old, before = run(data, "mkdata.py.bak-1108", old_flags, "before")
    new, after = run(data, flags=flags, name="after")
    assert old.returncode == new.returncode == 0, (old.stderr, new.stderr)
    assert before.read_bytes() == after.read_bytes()
    assert before.read_bytes()
    assert old.stdout == new.stdout
    assert old.stderr == new.stderr == b""


@pytest.mark.parametrize("quota", [1, 4, 1000])
def test_pool_only_hard_quota_and_provenance(data, quota):
    flags = ["--no-set", "--steps", "1", "--hard-skip", "2", "--ignore-kinds", "repeat,fu"]
    base, path = run(data, flags=flags, name="base")
    assert base.returncode == 0
    baseline = json.loads(path.read_text())
    result, path = run(data, flags=flags + ["--spk-neg", str(quota)])
    assert result.returncode == 0, result.stderr
    row = json.loads(path.read_text())
    assert row["docs"][1:12] == baseline["docs"][1:12]  # same 11 hard samples
    assert len(row["docs"]) == len(set(row["docs"])) == 16
    assert row["spk_quota"] == dict(requested=quota, capacity=min(quota, 4), realized=min(quota, 4))
    assert row["kinds"].count("hard") >= 11
    ids = row["round_ids"]
    assert set(ids) <= set(data[4]["fused"])
    assert not set(ids[1:]) & set(range(8))  # gold, support, ignored and hard-skipped
    sp = speakers_of(data[3])
    picked_sp = [sp[i] for i, kind in zip(ids, row["kinds"]) if kind == "spk"]
    assert set(picked_sp) <= {"Bob", "Alice"}
    if quota >= 4: assert "Alice" in picked_sp
    assert all(doc_user(doc).split("round-")[1].split(":")[0] == str(i) for doc, i in zip(row["docs"], ids))


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 7, 15])
def test_large_quota_small_k_and_no_eligible_speaker(data, k):
    # A huge requested quota must never produce a negative sample size.
    result, path = run(data, flags=["--no-set", "--steps", "2", "--K", str(k), "--spk-neg", "9999"])
    assert result.returncode == 0, result.stderr
    rows = list(map(json.loads, path.read_text().splitlines()))
    assert len(rows) == (0 if k < 4 else 2)
    for row in rows:
        assert len(row["docs"]) == k + 1
        assert row["spk_quota"]["capacity"] >= 0
    # No matching identities: fall back to safe classic negatives, without error.
    root, qfile, _, q, _ = data
    q["asker"] = "Nobody"
    for session in q["haystack_sessions"]:
        for t in session: t["content"] = "unattributed " + t["content"].split(": ", 1)[1]
    qfile.write_text(json.dumps([q]))
    result, path = run(data, flags=["--no-set", "--steps", "1", "--K", str(k), "--spk-neg", "100"])
    assert result.returncode == 0, result.stderr
    for row in map(json.loads, path.read_text().splitlines()): assert row["spk_quota"]["realized"] == 0


def test_filtering_selected_and_frozen_candidates(data):
    root = data[0]
    flags = ["--steps", "12", "--p-mixed", "1", "--spk-neg", "50", "--hard-skip", "2", "--ignore-kinds", "repeat,fu", "--fn-support", root / "fn.json"]
    result, path = run(data, flags=flags)
    assert result.returncode == 0, result.stderr
    rows = list(map(json.loads, path.read_text().splitlines()))
    assert rows and any(r["nsel"] for r in rows)
    for row in rows:
        assert not set(row["round_ids"][1:]) & set(row["selected"])
        assert not set(row["round_ids"][1:]) & {3, 4, 5, 6, 7, 68, 69, 70}
    result, path = run(data, flags=["--no-set", "--steps", "2", "--spk-neg", "4", "--cands", root / "cands.json", "--fn-support", root / "fn.json"])
    assert result.returncode == 0, result.stderr
    for row in map(json.loads, path.read_text().splitlines()):
        assert len(row["round_ids"]) == 16
        assert not set(row["round_ids"][1:]) & {68, 69, 70}


def test_truncated_pool_and_missing_teacher_scores(data):
    result, path = run(data, flags=["--no-set", "--steps", "3", "--spk-neg", "100", "--P", "75"])
    assert result.returncode == 0, result.stderr
    rows = list(map(json.loads, path.read_text().splitlines()))
    assert rows
    for row in rows:
        assert len(row["docs"]) == 16
        assert all(i < 75 for i in row["round_ids"])
    # Speaker negatives may be unscored but must still be in the retrieval pool.
    root, _, pooldir, _, pool = data
    pool["rr8"] = {str(i): (100 - i) / 101 for i in range(8)}
    (pooldir / "w1_q.json").write_text(json.dumps(pool))
    result, path = run(data, flags=["--no-set", "--steps", "1", "--spk-neg", "100"])
    assert result.returncode == 0, result.stderr
    row = json.loads(path.read_text())
    assert any(t is None for t, kind in zip(row["teacher"], row["kinds"]) if kind == "spk")
    assert set(row["round_ids"]) <= set(pool["fused"])


@pytest.mark.parametrize("flags", [["--spk-neg", "-1"], ["--spk-neg", "1", "--teacher-only"],
                                   ["--spk-neg", "1", "--neg-mix"], ["--spk-neg", "1", "--group-table", "unused"]])
def test_reject_invalid_or_incompatible_modes(data, flags):
    result, _ = run(data, flags=flags)
    assert result.returncode == 2
    assert b"--spk-neg" in result.stderr


def test_explicit_speaker_and_session_boundaries():
    q = dict(haystack_sessions=[[dict(role="assistant", content="alone", speaker="Bot"),
                                dict(role="user", content="question", speaker="A"),
                                dict(role="assistant", content="reply", speaker="Bot"),
                                dict(role="user", content="message", speaker="B")],
                               [dict(role="user", content="[Group 2] C: message")]])
    assert speakers_of(q) == ["Bot", "A", "B", "C"]
