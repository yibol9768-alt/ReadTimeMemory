"""D4 §3: seeded 50% aggregation/ordering rewrite; never changes evidence/pools.

Examples:
  python3 countfix.py --qfile env_ws4.json --ids ws4_tr.txt --dry-run
  python3 countfix.py --freeze D4_FREEZE.json --port 18300,18301 --out qtext.json
The output map includes selected failures with their original text. Gate rates
are reported only from supplied gate files; this script does not run evaluation.
"""
import argparse
import collections
import json
from pathlib import Path
import random
import re
import urllib.request
from d4data import read_json, write_json

MODEL = "Qwen3.8-27B-FP8"
PROMPT = """Rewrite the question below to remove hints about the NUMBER OF ANSWER ITEMS.
Remove explicit or spelled-out answer cardinalities, such as 'List 6', '6 items',
 'the 6 changes', 'five', or '六个'. Ask for the relevant items without revealing
how many there are. Keep the meaning, entities, constraints, date window, and any
requirement to order the answer. Preserve dates, identifiers and measurements
that define the scope; do not add facts, answers, examples, or new constraints.
Do not answer the question. Return ONLY a JSON object with one key "question"
and the rewritten question as its string value.

Original question:
{question}"""
# Protect dates, but conservatively reject remaining standalone numeric tokens
# and cardinal words. A conservative rejection keeps the original; it never
# silently accepts a possibly remaining count hint. Ordinal ordering words stay.
DATES = re.compile(r"\b\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?\b|"
                   r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b|"
                   r"\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?", re.I)
COUNTS = re.compile(r"(?<![\w])\d+(?:[.,]\d+)*(?![\w])|"
                    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|dozen|both|pair|couple)\b|"
                    r"[零〇一二两三四五六七八九十百千万]+\s*(?:个|项|条|件|种|次|位)", re.I)
MULTI = {"aggregation", "ordering"}


def count_hints(text):
    return COUNTS.findall(DATES.sub(" DATE ", text))


def chat(port, prompt, n=1500):
    """Same retry/content/finish handling as a6judge.chat; no judge imports/data."""
    last = None
    for attempt in range(4):
        body = dict(model=MODEL, messages=[dict(role="user", content=prompt)],
                    temperature=0, max_tokens=n * (2 if attempt >= 2 else 1),
                    chat_template_kwargs={"enable_thinking": False})
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                         json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as response:
                choice = json.loads(response.read())["choices"][0]
            content = choice["message"].get("content") or ""
            if content.strip() and choice.get("finish_reason") != "length":
                return content
            last = f"finish={choice.get('finish_reason')} empty={not content.strip()}"
        except Exception as exc:
            last = repr(exc)
    raise RuntimeError(f"no usable rewrite: {last}")


def gate_rates(path, ids):
    if not path:
        return None
    data = read_json(path)
    if isinstance(data, dict):
        data = data.get("rows", data)
        if isinstance(data, dict): data = [dict(v, qid=k) for k, v in data.items()]
    rows = [r for r in data if r.get("qid", r.get("question_id")) in ids]
    keys = ("blind", "suff", "del", "order", "passed")
    return dict(n=len(rows), missing=len(ids - {r.get("qid", r.get("question_id")) for r in rows}),
                means={key: sum(r[key] for r in rows if key in r) / sum(key in r for r in rows)
                       for key in keys if any(key in r for r in rows)})


def load_questions(a):
    keep = set(Path(a.ids).read_text().split()) if a.ids else None
    if a.freeze:
        freeze = read_json(a.freeze)
        keep = {q["qid"] for q in freeze["questions"]}
        base = Path(a.freeze).resolve().parent
        paths = [Path(s["qfile"]) for s in freeze["sources"]]
        paths = [p if p.is_absolute() else base / p for p in paths]
    else:
        paths = [Path(p) for p in a.qfile.split(",") if p]
    if not paths:
        raise ValueError("provide --qfile or --freeze")
    out = {}
    for path in paths:
        for q in read_json(path):
            qid = q["question_id"]
            if (keep is None or qid in keep) and q["question_type"] in MULTI:
                if qid in out: raise ValueError(f"duplicate question: {qid}")
                out[qid] = q
    return [out[qid] for qid in sorted(out)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qfile", default="", help="comma-separated training env files")
    ap.add_argument("--freeze", default="")
    ap.add_argument("--ids", default="")
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--port", default="", help="comma-separated existing OpenAI-compatible localhost ports")
    ap.add_argument("--out", default="qtext.json")
    ap.add_argument("--report", default="")
    ap.add_argument("--dry-run", action="store_true", help="print at most 3 prompts; no HTTP requests or output writes")
    ap.add_argument("--gates-before", default="")
    ap.add_argument("--gates-after", default="")
    a = ap.parse_args(argv)
    if a.qfile and a.freeze: ap.error("choose --qfile or --freeze")
    if a.ids and a.freeze: ap.error("--freeze already fixes the IDs")
    qs = load_questions(a)
    selected = random.Random(a.seed).sample(qs, len(qs) // 2)
    if a.dry_run:
        print(f"DRY RUN eligible={len(qs)} selected={len(selected)} seed={a.seed}; no requests")
        for q in selected[:3]:
            print(f"\n--- {q['question_id']} ---\n" + PROMPT.format(question=q["question"]))
        return
    ports = [p.strip() for p in a.port.split(",") if p.strip()]
    if not ports: ap.error("--port is required unless --dry-run")
    output, failures, changed, removed = {}, [], 0, 0
    originals_with_hints = sum(bool(count_hints(q["question"])) for q in selected)
    for index, q in enumerate(selected):
        qid, original = q["question_id"], q["question"]
        try:
            raw = chat(ports[index % len(ports)], PROMPT.format(question=original))
            data = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()))
            text = data["question"]
            if not isinstance(text, str) or not text.strip(): raise ValueError("empty/non-string question")
            text = text.strip()
            remaining = count_hints(text)
            if remaining: raise ValueError(f"remaining possible counts: {remaining}")
            # Date windows must remain byte-for-byte. Rejected rewrites preserve the original.
            if DATES.findall(original) != DATES.findall(text): raise ValueError("changed date scope")
            changed += text != original
            removed += bool(count_hints(original))
        except Exception as exc:
            text = original
            failures.append(dict(qid=qid, error=str(exc)))
        output[qid] = text
    ids = set(output)
    report = dict(model=MODEL, seed=a.seed, eligible=len(qs), selected=len(selected),
                  half_rule="floor(N/2), seeded sample of sorted aggregation/ordering IDs",
                  selected_ids=[q["question_id"] for q in selected],
                  selected_types=dict(collections.Counter(q["question_type"] for q in selected)),
                  failed=len(failures), failures=failures, changed=changed,
                  selected_with_count_hints=originals_with_hints, removed=removed,
                  removed_fraction_of_hint_questions=removed / originals_with_hints if originals_with_hints else 0,
                  removed_fraction_of_all_multi=removed / len(qs) if qs else 0,
                  gates_before=gate_rates(a.gates_before, ids), gates_after=gate_rates(a.gates_after, ids),
                  gate_note="not measured unless gate files supplied; no training or evaluation invoked",
                  candidate_pool="unchanged: original-question retrieval and 8B scores")
    write_json(a.out, output)
    write_json(a.report or a.out + ".report.json", report)
    print(f"eligible {len(qs)} selected {len(selected)} changed {changed} removed {removed} failures {len(failures)}")


if __name__ == "__main__":
    main()
