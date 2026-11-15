"""Structured-evidence test: per session in an existing context, the model (no question shown) writes dated, standalone
facts (relative times resolved against the session date). Writes ctx_<out>.jsonl = original ctx + notes sorted by date.
usage: python3 notes.py --src MOA --budget 4000 --out MOAN --ports P,.. """
import argparse, json, os, re, threading
from concurrent.futures import ThreadPoolExecutor
import run as R
PROMPT = ("Below is one past chat session between a user and an assistant, dated {date}.\n\n{text}\n\n"
          "List every fact about the USER stated in this session (events, purchases, amounts, counts, plans, possessions, "
          "people, preferences, and changes such as 'no longer', 'switched to', 'now'). Write each as one standalone "
          "sentence that starts with an ABSOLUTE date in YYYY/MM/DD: convert relative times ('yesterday', 'last week', "
          "'two months ago') using the session date; if no time is given, use the session date and add '(mentioned)'. "
          "Keep exact numbers and names. Output one fact per line, nothing else. If there are no user facts, output NONE.")

def chat(port, prompt):
    r = R.post(port, {"model": R.MODEL, "prompt": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                      "max_tokens": 1200, "temperature": 0, "stop": ["<|im_end|>"]})
    return r["choices"][0]["text"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--out", required=True); ap.add_argument("--ports", required=True); ap.add_argument("--par", type=int, default=96)
    a = ap.parse_args(); ports = a.ports.split(",")
    rows = [r for r in map(json.loads, open(os.path.join(R.D, f"ctx_{a.src}.jsonl"))) if r["budget"] == a.budget]
    jobs = []
    for qi, r in enumerate(rows):
        for si, block in enumerate(re.split(r"\n(?=### Session date: )", "\n" + r["ctx"])):
            m = re.match(r"### Session date: (.*)\n", block.strip("\n") + "\n")
            if m: jobs.append((qi, si, m.group(1).strip(), block.strip("\n").split("\n", 1)[1] if "\n" in block.strip("\n") else ""))
    res = {}; lock = threading.Lock()
    def one(ij):
        i, (qi, si, date, text) = ij
        out = chat(ports[i % len(ports)], PROMPT.format(date=date, text=text[:12000]))
        facts = [l.strip("-• ").strip() for l in out.splitlines() if re.match(r"^\s*[-•]?\s*\d{4}/\d{2}/\d{2}", l)]
        with lock: res[(qi, si)] = facts
    with ThreadPoolExecutor(a.par) as ex: list(ex.map(one, enumerate(jobs)))
    with open(os.path.join(R.D, f"ctx_{a.out}.jsonl"), "w") as f:
        for qi, r in enumerate(rows):
            facts = sorted({x for (q, s), fs in res.items() if q == qi for x in fs})
            note = "\n\nMemory notes (dated facts extracted from each session; times converted to absolute dates):\n" + "\n".join(facts)
            f.write(json.dumps({**r, "hist": a.out, "ctx": r["ctx"] + note, "n_notes": len(facts)}) + "\n")
    print("notes", a.out, len(rows), "sessions", len(jobs))

if __name__ == "__main__":
    main()
