"""Audits count consumed groups, preserve unknowns, and recover only safe provenance."""
from collections import Counter
import json

from audit_groups import Context, audit, load_context, main, markdown
from fmt import doc_str, query_str, round_prev, split_round
from k1 import rounds_of


def question(qid="w1_q", carrier="group", asker="Alice"):
    ts = [dict(role="user", content=f"[Group 1] {sp}: round {i}", speaker=sp, carrier=carrier)
          for i, sp in enumerate(("Alice", "Bob", "Chen"))]
    return dict(question_id=qid, question_type="direct", question="I (Alice) need the status.", asker=asker,
                haystack_sessions=[ts], haystack_dates=["2026-01-01"], gold_stages=["f1"])


def group(q, ids=(1, 0, 2), **kw):
    R, prev = rounds_of(q), round_prev(q)
    docs = [doc_str(R[i][1], *split_round(R[i][2]), prev=prev[i]) for i in ids]
    return dict(qid=q["question_id"], query=query_str(q["question"], "2026-11-01", []), docs=docs,
                **dict(label=0, kinds=["pos", "spk", "rand"], **kw))


def test_consumed_weighting_not_question_counts():
    q = question()
    q2 = question("w1_self")
    q2["gold_stages"] = ["f0"]
    unused = question("w2_unused", "narrative")
    rows = [group(q)] * 3 + [group(q2, ids=(0, 1, 2))]
    result = audit(rows, Context({x["question_id"]: x for x in (q, q2, unused)}))
    assert result["groups"] == 4
    assert result["counts"]["carrier"] == {"group": 4}
    assert result["counts"]["perspective"] == {"participant-other": 3, "self": 1}
    assert result["counts"]["positive_speaker_is_asker"] == {"no": 3, "yes": 1}
    assert result["counts"]["evidence_count"] == {"1": 4}
    assert result["negative_source"] == {"speaker": 4, "rand": 4}
    report = markdown(result, {"speaker": 2})
    assert "| participant-other | 3 | 75.00% |" in report
    assert "| speaker | 4 | 50.00% | 1.000 | 2 | 4 |" in report


def test_carriers_roles_world_names_and_unnamed():
    qs, rows = {}, []
    for i, carrier in enumerate(("group", "dialog", "dyadic", "narrative")):
        q = question(f"w{i}_q", carrier)
        if carrier == "dialog":
            q["question"] = "What did I prefer?"
            del q["asker"]
            for t in q["haystack_sessions"][0]:
                t.pop("speaker"); t["content"] = "a plain user message"
            q["haystack_sessions"][0] = [q["haystack_sessions"][0][0], dict(role="assistant", content="reply")]
            q["haystack_sessions"][0][0]["has_answer"] = True
        if carrier == "narrative":
            q["question"] = "What did Bob do?"; q.pop("asker")
        ids = (0,) if carrier == "dialog" else (1, 0, 2)
        row = group(q, ids=ids)
        qs[q["question_id"]] = q; rows.append(row)
    ctx = Context(qs, {"w1": dict(name="Alice")})
    result = audit(rows, ctx)
    assert result["counts"]["carrier"] == Counter({c: 1 for c in ("group", "user-assistant", "dyadic", "narrative")})
    assert result["counts"]["perspective"] == {"participant-other": 2, "self": 1, "unnamed": 1}
    assert result["counts"]["positive_speaker_is_asker"] == {"no": 2, "yes": 1, "unknown": 1}


def test_old_w20_shape_and_missing_fields():
    # Original W20 fields; no new provenance metadata required.
    row = dict(qid="w1_q", type="aggregation", query=query_str("List 3 items from January 01 to January 30.", "2026-11-01", []),
               docs=[doc_str("d", "fact", "reply"), doc_str("d", "other", "ack")], label=0,
               nsel=0, ncov=0, ngold=3, kinds=["pos", "hard"], teacher=[1, 0])
    result = audit([row, dict(docs=["no metadata"], label=0)])
    assert result["groups"] == 2
    assert result["counts"]["carrier"] == {"user-assistant": 1, "unknown": 1}
    assert result["counts"]["evidence_bucket"] == {"3-5": 1, "unknown": 1}
    assert result["counts"]["date_window"] == {"yes": 1, "unknown": 1}
    assert result["counts"]["list_n"] == {"yes": 1, "unknown": 1}
    assert result["negative_source"] == {"8b": 1}
    assert result["negative_type"] == {"unknown": 1}
    assert "unknown" in markdown(result)


def test_nonzero_label_source_types_and_quotas():
    q = question()
    row = group(q, ids=(2, 1, 0), neg_src=["struct", "rand"])
    row.update(label=1, kinds=["rand", "pos", "spk"], round_ids=[2, 1, 0])
    ctx = Context({q["question_id"]: q}, candidates={q["question_id"]: {"struct": {"2": "wrong_scope"}}})
    result = audit([row], ctx)
    assert result["counts"]["positive_speaker_is_asker"] == {"no": 1}
    assert result["negative_source"] == {"struct": 1, "rand": 1}
    assert result["negative_type"] == {"wrong_scope": 1, "unknown": 1}
    assert result["counts"]["negative_kind"] == {"rand": 1, "spk": 1}
    assert result["counts"]["source_quota"] == {"rand=1, struct=1": 1}
    assert "| wrong_scope | 1 | 50.00% | 1.000 | 2 | 1 |" in markdown(result, type_quotas={"wrong_scope": 2})
    # Both all-document and negatives-only arrays are accepted.
    row["neg_src"] = ["struct", "pos", "rand"]
    row["neg_types"] = ["wrong_event", "pos", "other_topic"]
    result = audit([row], ctx)
    assert result["negative_source"] == {"struct": 1, "rand": 1}
    assert result["negative_type"] == {"wrong_event": 1, "other_topic": 1}


def test_speaker_quota_and_origin_provenance():
    q = question()
    row = group(q, spk_quota=dict(requested=8, capacity=4, realized=1), spk_sources=["asker"])
    result = audit([row], Context({q["question_id"]: q}))
    assert result["counts"]["speaker_quota"] == {"requested=8, capacity=4, realized=1": 1}
    assert result["counts"]["speaker_negative_origin"] == {"asker": 1}
    assert result["counts"]["kind_quota"] == {"rand=1, spk=1": 1}


def test_teacher_only_and_missing_labels_are_not_negatives():
    rows = [dict(docs=["a", "b"], nolabel=True, teacher=[0, 1], kinds=["t", "t"]), dict(docs=["c"])]
    result = audit(rows)
    assert result["groups"] == 2 and result["labelled"] == result["negatives"] == 0
    assert result["counts"]["source_quota"] == {"none": 2}
    assert result["negative_source"] == {}
    assert result["counts"]["carrier"] == {"unknown": 2}


def test_consumed_question_not_instruction_or_original_shortcuts():
    q = question()
    q.update(question="List 5 items between 2025-01-01 and 2025-02-01.", scope="date")
    row = group(q)
    row["query"] = query_str("What changed?", "2026-11-01", [])
    result = audit([row], Context({q["question_id"]: q}))
    assert result["counts"]["list_n"] == {"no": 1}
    assert result["counts"]["date_window"] == {"no": 1}
    # Date in instruct alone and generic cardinalities are not List N or windows.
    row["query"] = query_str("Give three recommendations as of 2026-11-01.", "2026-11-01", [])
    result = audit([row])
    assert result["counts"]["list_n"] == {"no": 1}
    assert result["counts"]["date_window"] == {"no": 1}


def test_context_speaker_not_confused_with_document_speaker():
    doc = doc_str("d", "[Group 1] Bob: answer", "", prev="[Group 1] Alice: earlier")
    row = dict(docs=[doc], label=0, asker="Alice", query=query_str("I (Alice) need this.", "d", []))
    result = audit([row])
    assert result["counts"]["positive_speaker_is_asker"] == {"no": 1}
    assert result["counts"]["perspective"] == {"unknown": 1}  # participation not proven by a context mention


def test_assistant_only_evidence_and_unknown_attribution():
    q = dict(question_id="w1_q", question="What did I learn?", asker="Alice", gold_stages=["f1"],
             haystack_dates=["d"], haystack_sessions=[[
                 dict(role="user", content="question", speaker="Alice", has_answer=False),
                 dict(role="assistant", content="unique evidence", has_answer=True)]])
    row = group(q, ids=(0,))
    ctx = Context({q["question_id"]: q})
    result = audit([row], ctx)
    assert result["counts"]["positive_speaker_is_asker"] == {"no": 1}
    assert result["counts"]["perspective"] == {"participant-other": 1}
    for t in q["haystack_sessions"][0]: t.pop("has_answer")
    result = audit([row], Context({q["question_id"]: q}))
    assert result["counts"]["positive_speaker_is_asker"] == {"unknown": 1}


def test_ambiguous_or_stale_round_ids_remain_unknown():
    q = question()
    row = group(q)
    row["round_ids"] = [2, 0, 1]  # positive doc actually belongs to round 1
    result = audit([row], Context({q["question_id"]: q}))
    assert result["counts"]["carrier"] == {"unknown": 1}
    assert result["counts"]["positive_speaker_is_asker"] == {"unknown": 1}
    # Ambiguous exact matches with identical metadata can recover carrier but
    # cannot acquire a structural type via an invented unique round index.
    q["haystack_sessions"][0] = [dict(role="user", content="[Group 1] Bob: " + "long " * 40)] * 2
    row = group(q, ids=(0, 1), neg_src=["struct"])
    ctx = Context({q["question_id"]: q}, candidates={q["question_id"]: {"struct": {"0": "sibling"}}})
    result = audit([row], ctx)
    assert result["counts"]["carrier"] == {"group": 1}
    assert result["negative_type"] == {"unknown": 1}


def test_cli_jsonl_questions_worlds_and_empty(tmp_path, capsys):
    q = question()
    q.pop("asker")
    qfile, wfile, infile, outfile = [tmp_path / n for n in ("q.json", "w00001.json", "train.jsonl", "audit.md")]
    qfile.write_text(json.dumps([q])); wfile.write_text(json.dumps(dict(world=1, name="Alice")))
    infile.write_text(json.dumps(group(q)) + "\n\n" + json.dumps(group(q)) + "\n")
    main([str(infile), "--qfile", str(qfile), "--worlds", str(tmp_path), "--out", str(outfile), "--source-quotas", "speaker=2"])
    assert "Consumed groups: 2" in outfile.read_text()
    assert "| participant-other | 2 | 100.00% |" in outfile.read_text()
    ctx = load_context([qfile], [wfile], "")
    assert ctx.world("w1_q", {})["name"] == "Alice"
    infile.write_text("")
    main([str(infile)])
    assert "Consumed groups: 0" in capsys.readouterr().out
