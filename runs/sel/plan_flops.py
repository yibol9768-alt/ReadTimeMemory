"""Planned-path FLOPs of a training stage (SPEC_STAGE23 v3.2 5), computed before training from the frozen group files.
Tokenizes exactly as train.py encode(): query ids (cut to max-q as train.py does) + document ids (body without the Qwen
suffix, cut to max-d) + suffix ids, per document; padding excluded. FLOPs = 6 * params * train tokens * epochs
(+ 2 * params * reference-forward tokens * epochs when --w-base > 0). Groups are counted with their real sizes (min(len, group)).
usage (inside the vLLM image, CPU only): python3 plan_flops.py --model /models/qwen/Qwen3-Reranker-0.6B --data a.jsonl[,b.jsonl] --epochs 1 [--w-base 0.5]"""
import argparse, hashlib, json, os, random, sqlite3, sys, time


def token_cache_key(row, model, max_q, max_d, group, nsfx):
    """Hash only gtok's row inputs and tokenization settings, independent of row metadata."""
    payload = {"query": row["query"], "docs": row["docs"][:group], "model": model,
               "max_q": max_q, "max_d": max_d, "group": group, "nsfx": nsfx}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class TokenCountCache:
    """WAL readers coexist with short batched writes; never hold a lock while tokenizing."""
    def __init__(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.db = sqlite3.connect(os.path.join(directory, "tokens.sqlite3"), timeout=60)
        # Concurrent first open can make journal_mode fail immediately despite busy_timeout.
        deadline = time.monotonic() + 60
        while True:
            try:
                self.db.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                    self.db.close()
                    raise
                time.sleep(0.05)
        self.db.execute("CREATE TABLE IF NOT EXISTS tokens (key TEXT PRIMARY KEY, count INTEGER NOT NULL)")
        self.pending = {}
        self.hits = self.misses = 0

    def get(self, key, compute):
        if key in self.pending:
            self.hits += 1
            return self.pending[key]
        found = self.db.execute("SELECT count FROM tokens WHERE key = ?", (key,)).fetchone()
        if found is not None:
            self.hits += 1
            return found[0]
        count = compute()
        self.misses += 1
        self.pending[key] = count
        if len(self.pending) >= 128:
            self.flush()
        return count

    def flush(self):
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO tokens VALUES (?, ?)", self.pending.items())
        self.pending.clear()

    def close(self):
        try:
            self.flush()
        finally:
            self.db.close()


def main():
    from transformers import AutoTokenizer
    started = time.perf_counter()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-q", type=int, default=4000); ap.add_argument("--max-d", type=int, required=True); ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--w-base", type=float, default=0.0); ap.add_argument("--world", type=int, required=True); ap.add_argument("--accum", type=int, required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--params", type=int, default=0, help="parameter count (default: read from config via a CPU model load)")
    ap.add_argument("--token-cache-dir", default="/w/s23/cache/tok")
    ap.add_argument("--cache-stats", action="store_true", help="write cache hits/misses and timing to stderr")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    SFX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"; nsfx = len(tok(SFX, add_special_tokens=False)["input_ids"])
    tokenizer_seconds = time.perf_counter() - started
    if not a.params:
        from transformers import AutoModelForCausalLM
        a.params = sum(p.numel() for p in AutoModelForCausalLM.from_pretrained(a.model).parameters())
    def gtok(r):
        q = len(tok(r["query"], add_special_tokens=False)["input_ids"]); q = min(q, a.max_q)
        n = 0
        for d in r["docs"][: a.group]:
            body = d[: -len(SFX)] if d.endswith(SFX) else d
            n += q + min(len(tok(body, add_special_tokens=False)["input_ids"]), a.max_d) + nsfx
        return n
    # replicate train.py exactly: rows shuffled with Random(seed), split rows[rank::world], steps_total from the floor rule,
    # then per rank `order` drawn from the global random seeded with seed + rank (no other draws happen before it)
    rows = [json.loads(l) for p_ in a.data.split(",") for l in open(p_)]
    random.Random(a.seed).shuffle(rows)
    steps_total = int((len(rows) // a.world) * a.epochs) // a.accum
    persistent = TokenCountCache(a.token_cache_dir)
    cache = {}; train_tok = 0; groups = 0
    try:
        for rank in range(a.world):
            mine = rows[rank::a.world]; random.seed(a.seed + rank); order = []
            while len(order) < steps_total * a.accum: order += random.sample(range(len(mine)), len(mine))
            for k in range(steps_total * a.accum):
                j = (rank, order[k])
                if j not in cache:
                    row = mine[order[k]]
                    key = token_cache_key(row, a.model, a.max_q, a.max_d, a.group, nsfx)
                    cache[j] = persistent.get(key, lambda: gtok(row))
                train_tok += cache[j]; groups += 1
    finally:
        persistent.close()
    if a.cache_stats:
        print(json.dumps({"cache_hits": persistent.hits, "cache_misses": persistent.misses,
                          "tokenizer_seconds": tokenizer_seconds,
                          "elapsed_seconds": time.perf_counter() - started}), file=sys.stderr)
    ref_tok = train_tok if a.w_base > 0 else 0
    print(json.dumps({"data": a.data, "groups_processed": groups, "rows": len(rows), "steps": steps_total, "world": a.world, "accum": a.accum, "seed": a.seed, "epochs": a.epochs, "params": a.params, "train_tokens": int(train_tok),
                      "ref_forward_tokens": int(ref_tok), "planned_flops": 6 * a.params * train_tok + 2 * a.params * ref_tok}))


if __name__ == "__main__":
    main()
