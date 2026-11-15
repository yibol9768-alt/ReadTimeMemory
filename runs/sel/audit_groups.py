"""Audit the JSONL rows actually consumed by training, without model calls.

python audit_groups.py tr_w20.jsonl --qfile env_w20.json --worlds worlds/ --out audit_w20.md
--qfile and --worlds may be repeated; --cands optionally supplies D4 structural types.
Old rows need only docs/query/qid: absent or ambiguous metadata stays unknown.
Group shares count EVERY input row (including repeated questions), not source questions.
Carrier is the labelled positive's carrier; candidate carriers are reported separately.
Evidence count is ngold (available gold stages), falling back to question gold_stages;
it is not the number of positives sampled into the contrastive group.
Optional --source-quotas/--type-quotas accept e.g. '8b=3,rand=3,struct=5,ret=2,stu=2'.
"""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import re

from fmt import doc_str, round_prev, split_round
from k1 import rounds_of
from mkdata import speaker

UNKNOWN = "unknown"
CARRIERS = {
    "group": "group", "group_chat": "group", "group-chat": "group", "群聊": "group",
    "dialog": "user-assistant", "long": "user-assistant", "assistant": "user-assistant",
    "user_assistant": "user-assistant", "user-assistant": "user-assistant",
    "dyadic": "dyadic", "dyad": "dyadic", "two_person": "dyadic", "两人对聊": "dyadic",
    "narrative": "narrative", "narration": "narrative", "叙事": "narrative",
}
PERSPECTIVES = {"self": "self", "self_evidence": "self", "self_producer": "self",
                "participant_other": "participant-other", "participant-other": "participant-other",
                "other": "participant-other", "unnamed": "unnamed", "external": "unnamed"}
SELF = re.compile(r"\b(?:I|my|me|mine)\b", re.I)
LIST_N = re.compile(r"\blist\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)(?:\b|[- ])", re.I)
DATE = r"(?:\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,?\s+\d{4})?|\d{4}年\d{1,2}月(?:\d{1,2}日)?)"
DATE_WINDOW = re.compile(rf"(?:\b(?:between|from)\s+{DATE}\s+(?:and|to|through|until|[-–—])\s+{DATE}|{DATE}\s*(?:to|through|until|[-–—]|至|到)\s*{DATE})", re.I)


def first(*values):
    return next((v for v in values if v is not None), None)


def aligned_messages(q):
    """The same boundary rule as k1, including lone assistants and session ends."""
    out = []
    for sess in q.get("haystack_sessions", []):
        i = 0
        while i < len(sess):
            paired = sess[i]["role"] == "user" and i + 1 < len(sess) and sess[i + 1]["role"] == "assistant"
            out.append(sess[i:i + (2 if paired else 1)])
            i += 2 if paired else 1
    return out


def question_text(row, q):
    # The consumed query takes precedence over original question text (D4 rewrites).
    m = re.search(r"<Query>: (.*?)\n<Selected>:", row.get("query", ""), re.S)
    return m.group(1) if m else row.get("question", q.get("question"))


def doc_user(doc):
    # Ignore previous-message context and the instruction's first-person pronouns.
    m = re.match(r"<Document>: \([^)]*\) (?:\[follow-up to: [^\n]*\]\n)?User: (.*?)(?:\nAssistant:|<\|im_end\|>)", doc, re.S)
    return m.group(1) if m else None


def bool_name(value):
    return "yes" if value is True else "no" if value is False else UNKNOWN


class Context:
    def __init__(self, questions=None, worlds=None, candidates=None):
        self.questions = questions or {}
        self.worlds = worlds or {}
        self.candidates = candidates or {}

    def world(self, qid, q):
        wid = str(first(q.get("world_id"), q.get("world"), qid.split("_")[0]))
        if wid in self.worlds: return self.worlds[wid]
        if re.fullmatch(r"w?\d+", wid): return self.worlds.get(f"w{int(wid.lstrip('w'))}", {})
        return {}

    @lru_cache(maxsize=8)
    def rounds(self, qid):
        q = self.questions.get(qid, {})
        if not q.get("haystack_sessions") or "haystack_dates" not in q: return [], {}
        R, prev = rounds_of(q), round_prev(q)
        msgs = aligned_messages(q)
        w = self.world(qid, q)
        wr = w.get("rounds", [])
        out, by_doc = [], defaultdict(list)
        for i, (r, ts) in enumerate(zip(R, msgs)):
            user, assistant = split_round(r[2])
            rendered = doc_str(r[1], user, assistant, prev=prev[i])
            # World round metadata is usable only when aligned, never by truncating zip.
            wm = wr[i] if len(wr) == len(R) else {}
            carrier = first(ts[0].get("carrier"), wm.get("carrier"), q.get("carrier"), w.get("carrier"))
            carrier = CARRIERS.get(carrier, UNKNOWN)
            sp = first(ts[0].get("speaker"), wm.get("speaker"), speaker(r[2]))
            if carrier == UNKNOWN and speaker(r[2]): carrier = "group"
            # Role structure reliably identifies assistant chats; no inference of
            # narrative vs two-person chats from bare unannotated user messages.
            if carrier == UNKNOWN and len(ts) == 2: carrier = "user-assistant"
            if sp is None and carrier == "user-assistant": sp = first(q.get("user_name"), w.get("name"))
            participants = {sp} if sp else set()
            # A paired positive may carry assistant-only evidence. Prefer actual
            # gold message attribution; absent attribution must not imply "user".
            if len(ts) == 2:
                ap = ts[1].get("speaker") or "assistant"
                participants.add(ap)
                evidence_speakers = {sp if t is ts[0] else ap for t in ts if t.get("has_answer") in (True, 1, "True")}
                sp = next(iter(evidence_speakers)) if len(evidence_speakers) == 1 else None
            out.append(dict(carrier=carrier, speaker=sp, participants=participants, doc=rendered))
            by_doc[rendered].append(i)
        return out, by_doc

    def resolve(self, row, qid, j):
        R, by_doc = self.rounds(qid)
        ids = row.get("round_ids") or []
        explicit = ids[j] if j < len(ids) else row.get("positive") if j == row.get("label") else None
        doc = row["docs"][j]
        if isinstance(explicit, int) and 0 <= explicit < len(R):
            # Do not silently trust stale ids from another world/rendering.
            if R[explicit]["doc"] == doc: return R[explicit], explicit
            return dict(carrier=UNKNOWN, speaker=None), None
        matches = by_doc.get(doc, [])
        if matches:
            # Identical documents can correspond to several rounds: only recover
            # metadata when every match agrees, never invent a unique round id.
            carriers = {R[i]["carrier"] for i in matches}; speakers = {R[i]["speaker"] for i in matches}
            return dict(carrier=next(iter(carriers)) if len(carriers) == 1 else UNKNOWN,
                        speaker=next(iter(speakers)) if len(speakers) == 1 else None), matches[0] if len(matches) == 1 else None
        user = doc_user(doc)
        sp = speaker("User: " + user) if user is not None else None
        am = re.search(r"\nAssistant: (.*?)<\|im_end\|>", doc, re.S)
        carrier = "group" if sp else "user-assistant" if am and am.group(1).strip() else UNKNOWN
        return dict(carrier=carrier, speaker=sp), None


def labels(row, name, neg_indices, n_docs):
    values = row.get(name)
    if not isinstance(values, list): return [UNKNOWN] * len(neg_indices)
    if len(values) == n_docs: return [values[j] or UNKNOWN for j in neg_indices]
    if len(values) == len(neg_indices): return [v or UNKNOWN for v in values]
    return [UNKNOWN] * len(neg_indices)


def audit(rows, context=None):
    context = context or Context()
    counts = defaultdict(Counter)
    source_totals, type_totals = Counter(), Counter()
    source_groups, type_groups = [], []
    n = labelled = nneg = 0
    for row in rows:
        n += 1
        qid = row.get("qid", row.get("question_id", ""))
        q = context.questions.get(qid, {})
        w = context.world(qid, q)
        docs = row.get("docs", [])
        label = row.get("label")
        valid = not row.get("nolabel", False) and isinstance(label, int) and 0 <= label < len(docs)
        labelled += int(valid)
        counts["label_status"]["labelled" if valid else "unlabelled"] += 1
        text = question_text(row, q)
        counts["question_type"][first(row.get("type"), q.get("question_type"), UNKNOWN)] += 1
        evidence = first(row.get("ngold"), len(q["gold_stages"]) if "gold_stages" in q else None)
        counts["evidence_count"][str(evidence) if evidence is not None else UNKNOWN] += 1
        bucket = UNKNOWN if not isinstance(evidence, int) else "0" if evidence == 0 else "1" if evidence == 1 else "2" if evidence == 2 else "3-5" if evidence <= 5 else "6-8" if evidence <= 8 else "9+"
        counts["evidence_bucket"][bucket] += 1
        counts["list_n"][bool_name(bool(LIST_N.search(text)) if text is not None else None)] += 1
        # Source scope must not reintroduce a window removed in a consumed rewrite.
        consumed = "<Query>: " in row.get("query", "") or row.get("question") is not None
        dated = bool(DATE_WINDOW.search(text)) if text is not None else None
        dated = first(row.get("date_window"), dated if consumed else None,
                      q.get("date_window"), True if q.get("scope") == "date" else dated)
        if dated is not None and not isinstance(dated, bool): dated = bool(dated)
        counts["date_window"][bool_name(dated)] += 1
        resolved = [context.resolve(row, qid, j) for j in range(len(docs))]
        for info, _ in resolved: counts["candidate_carrier"][info["carrier"]] += 1
        pos = resolved[label][0] if valid else dict(carrier=UNKNOWN, speaker=None)
        counts["carrier"][pos["carrier"]] += 1
        asker = first(row.get("asker"), q.get("asker"))
        if asker is None and text and SELF.search(text): asker = first(q.get("user_name"), w.get("name"))
        relation = UNKNOWN if not valid or not asker or not pos["speaker"] else "yes" if asker == pos["speaker"] else "no"
        counts["positive_speaker_is_asker"][relation] += 1
        perspective = PERSPECTIVES.get(first(row.get("perspective"), q.get("perspective")), UNKNOWN)
        if perspective == UNKNOWN:
            if relation == "yes": perspective = "self"
            elif relation == "no":
                R, _ = context.rounds(qid)
                participants = {sp for r in R for sp in r.get("participants", [])} | {info["speaker"] for info, _ in resolved if info["speaker"]}
                if asker in participants or asker in w.get("team", []): perspective = "participant-other"
            elif text is not None and not asker and not SELF.search(text): perspective = "unnamed"
        counts["perspective"][perspective] += 1
        # Teacher-only candidates have no positive/negative semantics.
        neg_indices = [j for j in range(len(docs)) if j != label] if valid else []
        kinds = labels(row, "kinds", neg_indices, len(docs))
        sources = labels(row, "neg_src", neg_indices, len(docs))
        types = labels(row, "neg_types", neg_indices, len(docs))
        entry = context.candidates.get(qid, {})
        counts["negative_count"][str(len(neg_indices)) if valid else UNKNOWN] += 1
        sq = row.get("spk_quota")
        counts["speaker_quota"][", ".join(f"{k}={sq.get(k, UNKNOWN)}" for k in ("requested", "capacity", "realized")) if isinstance(sq, dict) else UNKNOWN] += 1
        origins = row.get("spk_sources", [])
        if len(origins) != kinds.count("spk"): origins = [UNKNOWN] * kinds.count("spk")
        counts["speaker_negative_origin"].update(origins)
        for k, j in enumerate(neg_indices):
            counts["negative_carrier"][resolved[j][0]["carrier"]] += 1
            counts["negative_kind"][kinds[k]] += 1
            if sources[k] == UNKNOWN:
                sources[k] = {"hard": "8b", "red": "redundant", "spk": "speaker", "rand": "rand"}.get(kinds[k], UNKNOWN)
            if types[k] == UNKNOWN and sources[k] == "struct":
                rid = resolved[j][1]
                struct = entry.get("struct", {})
                if isinstance(struct, dict): types[k] = struct.get(str(rid), UNKNOWN)
        sc, tc = Counter(sources), Counter(types)
        source_totals.update(sc); type_totals.update(tc)
        source_groups.append(sc); type_groups.append(tc)
        nneg += len(neg_indices)
        for name, counter in (("source_quota", sc), ("type_quota", tc), ("kind_quota", Counter(kinds))):
            signature = ", ".join(f"{k}={v}" for k, v in sorted(counter.items())) or "none"
            counts[name][signature] += 1
    return dict(groups=n, labelled=labelled, negatives=nneg, counts=dict(counts),
                negative_source=source_totals, negative_type=type_totals,
                source_groups=source_groups, type_groups=type_groups)


def markdown(result, source_quotas=None, type_quotas=None):
    def esc(value): return str(value).replace("|", "\\|").replace("\n", " ")
    n = result["groups"]
    lines = ["# Training group audit", "", f"Consumed groups: {n}; labelled: {result['labelled']}; negatives: {result['negatives']}.", "",
             "Each input row has weight 1; repeated questions and boosted groups remain repeated. Group carrier and perspective refer to the sampled positive. Evidence count = available gold stages (ngold), not sampled positives. Unknown values remain in denominators. Unlabelled teacher groups have no negative labels.", "",
             "Date-window detection uses explicit metadata or two date endpoints in the consumed question; the instruction's question date is excluded. List N is a surface-pattern statistic, not a semantic leakage verdict.", ""]
    for name, counter in result["counts"].items():
        total = sum(counter.values())
        lines += [f"## {name}", "", "| Value | Count | Share |", "| --- | ---: | ---: |"]
        for key, count in sorted(counter.items()): lines.append(f"| {esc(key)} | {count} | {count / total:.2%} |")
        lines.append("")
    for name, quotas in (("source", source_quotas or {}), ("type", type_quotas or {})):
        counter = result[f"negative_{name}"]
        groups = result[f"{name}_groups"]
        lines += [f"## Negative {name} quota totals", "", "| Value | Negatives | Share of negatives | Mean / consumed group | Target / group | Groups below target |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for key in sorted(counter.keys() | quotas.keys()):
            target = quotas.get(key)
            below = sum(g[key] < target for g in groups) if target is not None else UNKNOWN
            lines.append(f"| {esc(key)} | {counter[key]} | {counter[key] / result['negatives'] if result['negatives'] else 0:.2%} | {counter[key] / n if n else 0:.3f} | {target if target is not None else UNKNOWN} | {below} |")
        lines.append("")
    return "\n".join(lines)


def json_rows(paths):
    for path in paths:
        with open(path) as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip(): continue
                try: yield json.loads(line)
                except ValueError as exc: raise ValueError(f"{path}:{line_no}: {exc}") from exc


def load_context(qfiles, world_paths, cands):
    qs, worlds = {}, {}
    for path in qfiles:
        data = json.loads(Path(path).read_text())
        for q in data if isinstance(data, list) else [data]:
            qid = q["question_id"]
            if qid in qs: raise ValueError(f"duplicate question_id: {qid}")
            qs[qid] = q
    for source in world_paths:
        p = Path(source)
        for path in sorted(p.glob("w*.json")) if p.is_dir() else [p]:
            w = json.loads(path.read_text())
            wid = first(w.get("world_id"), w.get("world"), w.get("id"), path.stem)
            key = f"w{int(str(wid).lstrip('w'))}" if re.fullmatch(r"w?\d+", str(wid)) else str(wid)
            if key in worlds: raise ValueError(f"duplicate world: {key}")
            worlds[key] = w
    entries = json.loads(Path(cands).read_text()) if cands else {}
    return Context(qs, worlds, entries.get("q", entries))


def parse_quotas(text):
    result = {}
    for pair in text.split(",") if text else []:
        key, value = pair.split("=", 1)
        result[key.strip()] = int(value)
        if result[key.strip()] < 0: raise ValueError("quotas must be nonnegative")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("groups", nargs="+")
    ap.add_argument("--qfile", action="append", default=[])
    ap.add_argument("--worlds", action="append", default=[])
    ap.add_argument("--cands", default="")
    ap.add_argument("--source-quotas", default="")
    ap.add_argument("--type-quotas", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    context = load_context(a.qfile, a.worlds, a.cands)
    report = markdown(audit(json_rows(a.groups), context), parse_quotas(a.source_quotas), parse_quotas(a.type_quotas))
    if a.out: Path(a.out).write_text(report, encoding="utf-8")
    else: print(report)


if __name__ == "__main__":
    main()
