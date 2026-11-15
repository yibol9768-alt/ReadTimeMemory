"""CPU-only question identity memo regressions using real file metadata."""
import json
import os
from unittest.mock import patch

import pytest
import s23


def write_gold(path, qid):
    path.write_text(json.dumps({"qid": qid, "query": "query", "docs": ["doc"] * 5,
                                "teacher": [0.5] * 5, "label": 0, "nsel": 0}) + "\n")


@pytest.mark.parametrize("change", ["mtime", "size"])
def test_validators_share_memo_and_reload_changed_metadata(tmp_path, monkeypatch, change):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(s23, "BENCH", ["beam"])
    monkeypatch.setattr(s23, "_QUESTION_IDS_CACHE", {})
    qfile = tmp_path / "q_train_beam.json"
    freeze = tmp_path / "DATA_FREEZE_STAGE23.json"
    source = tmp_path / "source.jsonl"
    qfile.write_text(json.dumps([{"question_id": "a"}]))
    freeze.write_text(json.dumps({"bench": {"beam": {"train": ["a"], "inner": ["inner"]}}}))
    write_gold(source, "a")
    params = {"all_data": True, "qfile": qfile.name, "kind": "gold"}

    with patch.object(s23.json, "load", wraps=json.load) as load:
        for _ in range(2):
            s23.validate_train_files()
            assert s23.validate_source(source, params) == 1
        assert s23._question_ids(qfile) == ["a"]  # Absolute/relative aliases share a key.
        count = lambda: sum(os.fspath(c.args[0].name).endswith(qfile.name) for c in load.call_args_list)
        assert count() == 1
        before = qfile.stat()
        qid = "b" if change == "mtime" else "longer"
        qfile.write_text(json.dumps([{"question_id": qid}]))
        mtime = before.st_mtime_ns + 1_000_000 if change == "mtime" else before.st_mtime_ns
        os.utime(qfile, ns=(before.st_atime_ns, mtime))
        after = qfile.stat()
        assert (after.st_size == before.st_size) == (change == "mtime")
        assert (after.st_mtime_ns == before.st_mtime_ns) == (change == "size")
        with pytest.raises(ValueError, match="non-training qid"):
            s23.validate_source(source, params)
        with pytest.raises(ValueError, match="identities"):
            s23.validate_train_files()
        freeze.write_text(json.dumps({"bench": {"beam": {"train": [qid], "inner": ["inner"]}}}))
        write_gold(source, qid)
        s23.validate_train_files()
        assert s23.validate_source(source, params) == 1
        assert count() == 2
        assert list(s23._QUESTION_IDS_CACHE) == [(str(qfile), after.st_size, after.st_mtime_ns)]


@pytest.mark.parametrize("data,error", [("[", json.JSONDecodeError), ('[{}]', KeyError)])
def test_bad_questions_raise_on_every_call_without_caching(tmp_path, monkeypatch, data, error):
    monkeypatch.setattr(s23, "_QUESTION_IDS_CACHE", {})
    path = tmp_path / "q_train_bad.json"
    path.write_text(data)
    for _ in range(2):
        with pytest.raises(error):
            s23._question_ids(path)
    assert not s23._QUESTION_IDS_CACHE


def test_freeze_is_revalidated_on_question_memo_hit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(s23, "BENCH", ["beam"])
    monkeypatch.setattr(s23, "_QUESTION_IDS_CACHE", {})
    (tmp_path / "q_train_beam.json").write_text('[{"question_id":"a"}]')
    freeze = tmp_path / "DATA_FREEZE_STAGE23.json"
    freeze.write_text('{"bench":{"beam":{"train":["a"],"inner":["b"]}}}')
    s23.validate_train_files()
    freeze.write_text('{"bench":{"beam":{"train":["a"],"inner":["a"]}}}')
    with pytest.raises(ValueError, match="identities"):
        s23.validate_train_files()


def test_malformed_identity_exception_order_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(s23, "BENCH", ["beam"])
    monkeypatch.setattr(s23, "_QUESTION_IDS_CACHE", {})
    (tmp_path / "q_train_beam.json").write_text('[{"question_id":[]},{}]')
    (tmp_path / "DATA_FREEZE_STAGE23.json").write_text('{"bench":{}}')
    source = tmp_path / "source.jsonl"
    write_gold(source, "a")
    for _ in range(2):
        with pytest.raises(TypeError, match="unhashable"):
            s23.validate_source(source, {"all_data": True, "qfile": "q_train_beam.json", "kind": "gold"})
        with pytest.raises(KeyError, match="question_id"):
            s23.validate_train_files()
    assert not s23._QUESTION_IDS_CACHE
