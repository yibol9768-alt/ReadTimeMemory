"""Runs ON the H20 (copied to ~/v3_emb/). LycheeMemory-V2-style segmentation without an LLM: reads
{"convs": {name: [[session_no, turn_text], ...]}} from stdin, embeds every turn with qwen3-embedding-8b, and
writes {name: [segment start indices]} to stdout. A segment closes when
  - the session changes (sessions are days apart), or
  - the boundary probability p > 0.5, where p = sigmoid(2.5*z_shift + 2.0*(len/MAX_LEN - 1) + 1.5*z_cohesion_drop)
    z_shift          = how far the new turn is from the current segment's centroid (z-score within the conversation)
    z_cohesion_drop  = how much the segment's mean similarity to its centroid would fall by adding the turn
    len/MAX_LEN      = length pressure
and a segment has at least MIN_LEN turns."""
import json, math, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remote_rank import embed, CACHE          # same embedding cache as the ranking script

MIN_LEN, MAX_LEN = 3, 14


def segment(sess, V):
    n = len(V)
    shift = np.array([0.0] + [1 - float(V[i] @ V[i - 1]) for i in range(1, n)])
    mu, sd = shift[1:].mean(), shift[1:].std() + 1e-6
    starts, cur = [0], [0]
    for i in range(1, n):
        if sess[i] != sess[i - 1]:
            starts.append(i); cur = [i]; continue
        c = V[cur].mean(0); c /= np.linalg.norm(c)
        z_shift = ((1 - float(V[i] @ c)) - mu) / sd
        coh_now = float((V[cur] @ c).mean())
        c2 = V[cur + [i]].mean(0); c2 /= np.linalg.norm(c2)
        coh_new = float((V[cur + [i]] @ c2).mean())
        z_drop = (coh_now - coh_new) / sd
        p = 1 / (1 + math.exp(-(2.5 * z_shift + 2.0 * (len(cur) / MAX_LEN - 1) + 1.5 * z_drop)))
        if len(cur) >= MIN_LEN and p > 0.5:
            starts.append(i); cur = [i]
        else:
            cur.append(i)
    return starts


def main():
    job = json.load(sys.stdin)
    store = dict(np.load(CACHE)) if os.path.exists(CACHE) else {}
    out = {}
    for name, turns in job["convs"].items():
        V = embed([t for s, t in turns], store)
        out[name] = segment([s for s, t in turns], V)
    np.savez(CACHE, **store)
    json.dump(out, sys.stdout)


main()
