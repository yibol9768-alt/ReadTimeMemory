"""Answer-in-context recall at equal budget: UI store query vs BM25 over raw state chunks (same keywords)."""
import json, re, math, collections, sys
sys.path.insert(0, '.')
from mem.uistore import UIIndex, keywords
qs = [json.loads(l) for l in open('runs/lmev2/data/questions.jsonl')]
hs = json.load(open('runs/lmev2/data/haystacks/lme_v2_small.json'))
need = set(sum(hs.values(), []))
B = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
chunks = {'web': [], 'enterprise': []}
for l in open('runs/lmev2/data/trajectories.jsonl'):
    t = json.loads(l)
    if t['id'] not in need: continue
    for s in t['states']:
        body = f"[{t['goal'][:150]}] step {s['state_index']} {s['url']}\naction: {s['action']}\nthought: {s['thought']}\n{s['accessibility_tree'] or ''}"
        for i in range(0, len(body), 4000):
            chunks[t['domain']].append(body[i:i + 4000])
def bm25_index(docs):
    toks = [collections.Counter(re.findall(r"\w+", d.lower())) for d in docs]
    df = collections.Counter(w for t in toks for w in t)
    avg = sum(sum(t.values()) for t in toks) / len(toks)
    return toks, df, avg
idx = {d: bm25_index(c) for d, c in chunks.items()}
def bm25(d, q):
    toks, df, avg = idx[d]; N = len(toks)
    qt = [w for k in keywords(q) for w in re.findall(r"\w+", k.lower())]
    sc = []
    for i, t in enumerate(toks):
        L = sum(t.values()); s = 0
        for w in qt:
            if w in t:
                s += math.log(1 + (N - df[w] + .5) / (df[w] + .5)) * t[w] * 2.2 / (t[w] + 1.2 * (.25 + .75 * L / avg))
        sc.append(s)
    order = sorted(range(N), key=lambda i: -sc[i])
    out, used = [], 0
    for i in order:
        if used + len(chunks[d][i]) > B: break
        out.append(chunks[d][i]); used += len(chunks[d][i])
    return "\n".join(out)
ui = {d: UIIndex(f'runs/lmev2/store_small_{d}') for d in ['web', 'enterprise']}
r = collections.defaultdict(lambda: [0, 0, 0])
for q in qs:
    if 'abs' in q['question_type'] or 'gotchas' in q['question_type'] or q['eval_function'].startswith('mc_'): continue
    gold = [g.strip().lower() for g in re.split(r'[;,]', str(q['answer'])) if g.strip()]
    a = ui[q['domain']].query(q['question'], budget_chars=B)[0].lower()
    b = bm25(q['domain'], q['question']).lower()
    x = r[q['question_type']]; x[0] += 1; x[1] += all(g in a for g in gold); x[2] += all(g in b for g in gold)
print(f"budget {B} chars:", {t: f"n={v[0]} uistore={v[1]} bm25={v[2]}" for t, v in r.items()})
