"""CPU-only ordered concurrency regressions; all HTTP calls are mocked."""
from collections import Counter
import json
import re
import threading
from unittest.mock import patch

import pytest
import inner_rank


@pytest.fixture
def ranking_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    questions = [{"question_id": f"q{n:02}", "question": f"question-{n:02} café?",
                  "question_date": "2026-01-02", "haystack_dates": ["2026-01-01"],
                  "haystack_sessions": [[{"role": "user", "content": f"round-{i} café",
                                           "has_answer": i == 3} for i in range(67)]]}
                 for n in range(18)]
    (tmp_path / "DATA_FREEZE_STAGE23.json").write_text(json.dumps({"bench": {
        b: {"inner": [q["question_id"] for q in questions]} for b in ("beam", "lme")}}))
    for b in ("beam", "lme"):
        (tmp_path / f"q_inner_{b}.json").write_text(json.dumps(questions))
    pools = tmp_path / "pool_s23"
    pools.mkdir()
    for q in questions:
        (pools / f"{q['question_id']}.json").write_text(json.dumps({
            "fused": list(reversed(range(66))), "rr8": {str(i): (i % 7) / 10 for i in range(66)}}))
    return tmp_path, questions


@pytest.mark.parametrize("style,context,weight", [
    ("selector", False, 0.0), ("selector", True, 0.5),
    ("official", True, 1.0), ("plain", True, 0.5)])
def test_parallel_and_serial_files_and_requests_are_identical(ranking_files, style, context, weight):
    directory, questions = ranking_files
    requests = {}
    for par in (1, 16):
        barrier = threading.Barrier(16, timeout=10)
        second_finished = threading.Event()
        lock = threading.Lock()
        calls, batches, completions = [], Counter(), []
        active = peak = 0

        def post(url, body):
            nonlocal active, peak
            assert url == "http://127.0.0.1:mock/v1/rerank"
            qid = int(re.search(r"question-(\d+)", body["query"]).group(1))
            with lock:
                calls.append(json.dumps(body, sort_keys=True))
                batches[qid] += 1
                first_batch = batches[qid] == 1
                active += 1
                peak = max(peak, active)
            if par == 16 and first_batch and qid < 16:
                barrier.wait()
                if qid == 0:
                    assert second_finished.wait(10)
            result = {"results": [{"index": i, "relevance_score": sum(d.encode()) % 7 / 10}
                                   for i, d in reversed(list(enumerate(body["documents"])))]}
            with lock:
                active -= 1
                if len(body["documents"]) == 2:
                    completions.append(qid)
                    if qid == 1: second_finished.set()
            return result

        with patch.object(inner_rank, "_post", side_effect=post):
            # A second benchmark also exercises executor shutdown and loop closures.
            inner_rank.rank_inner(f"p{par}", ["beam", "lme"], port="mock", style=style,
                                  neighbour_context=context, rrf_weight=weight, par=par)
        assert peak == par
        assert len(calls) == len(questions) * 2 * 2
        if par == 16:
            assert completions.index(1) < completions.index(0)
        requests[par] = Counter(calls)
    assert requests[1] == requests[16]
    for bench in ("beam", "lme"):
        serial = (directory / f"ctx_p1_inner_{bench}.jsonl").read_bytes()
        assert serial == (directory / f"ctx_p16_inner_{bench}.jsonl").read_bytes()
        rows = [json.loads(line) for line in serial.splitlines()]
        assert [(r["qid"], r["budget"]) for r in rows] == [
            (q["question_id"], b) for q in questions for b in inner_rank.B]


def test_default_env_and_keyword_workers_and_cached_scores(ranking_files, monkeypatch):
    directory, _ = ranking_files
    real_executor = inner_rank.ThreadPoolExecutor
    monkeypatch.delenv("INNER_RANK_PAR", raising=False)
    with patch.object(inner_rank, "ThreadPoolExecutor", wraps=real_executor) as executor, \
            patch.object(inner_rank, "_post", side_effect=AssertionError("cached score HTTP call")):
        inner_rank.rank_inner("default", ["beam"], score_key="rr8")
        assert executor.call_args.kwargs["max_workers"] == 16
        monkeypatch.setenv("INNER_RANK_PAR", "2")
        inner_rank.rank_inner("env", ["beam"], score_key="rr8")
        assert executor.call_args.kwargs["max_workers"] == 2
        executor.reset_mock()
        inner_rank.rank_inner("serial", ["beam"], score_key="rr8", par=1)
        executor.assert_not_called()
    assert len({(directory / f"ctx_{tag}_inner_beam.jsonl").read_bytes()
                for tag in ("default", "env", "serial")}) == 1


@pytest.mark.parametrize("par", [1, 16])
def test_http_validation_exceptions_propagate(ranking_files, par):
    with patch.object(inner_rank, "_post", return_value={"results": []}):
        with pytest.raises(ValueError, match="incomplete rerank result"):
            inner_rank.rank_inner("bad", ["beam"], port="mock", par=par)
