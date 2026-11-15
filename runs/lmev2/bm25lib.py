"""BM25 over raw state chunks (4000 chars: goal, step, URL, action, thought, accessibility tree) of a Small haystack."""
import json, math, re, collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mem.uistore import expanded_keywords as keywords
D = os.path.dirname(os.path.abspath(__file__))


class BM25:
    def __init__(self, domain):
        hs = json.load(open(os.path.join(D, "data/haystacks/lme_v2_small.json")))
        need = set(sum(hs.values(), []))
        self.chunks = []
        for l in open(os.path.join(D, "data/trajectories.jsonl")):
            t = json.loads(l)
            if t["id"] not in need or t["domain"] != domain:
                continue
            for s in t["states"]:
                body = (f"[{t['goal'][:150]} | {t['outcome']}] step {s['state_index']} {s['url']}\naction: {s['action']}\n"
                        f"thought: {s['thought']}\n{s['accessibility_tree'] or ''}")
                for i in range(0, len(body), 4000):
                    self.chunks.append(body[i:i + 4000])
        self.toks = [collections.Counter(re.findall(r"\w+", d.lower())) for d in self.chunks]
        self.df = collections.Counter(w for t in self.toks for w in t)
        self.avg = sum(sum(t.values()) for t in self.toks) / len(self.toks)

    def query(self, q, budget):
        N = len(self.toks)
        qt = [w for k in keywords(q) for w in re.findall(r"\w+", k.lower())]
        sc = []
        for i, t in enumerate(self.toks):
            L = sum(t.values()); s = 0.0
            for w in qt:
                if w in t:
                    idf = math.log(1 + (N - self.df[w] + .5) / (self.df[w] + .5))
                    s += idf * t[w] * 2.2 / (t[w] + 1.2 * (.25 + .75 * L / self.avg))
            sc.append(s)
        out, used = [], 0
        for i in sorted(range(N), key=lambda i: -sc[i]):
            if used + len(self.chunks[i]) > budget:
                break
            out.append(self.chunks[i]); used += len(self.chunks[i])
        return "\n\n".join(out)
