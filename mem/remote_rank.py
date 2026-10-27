"""Runs ON the H20 (copied to ~/v3_emb/). Reads {"docs": {group: [text, ...]}, "queries": [[group, text], ...]}
from stdin, embeds with the local qwen3-embedding-8b (port 8002), and writes {"group\\ttext": [doc index, ...]}
(every doc, best first) to stdout. Document vectors are cached in ~/v3_emb/cache.npz between calls, so only
rankings, never vectors, cross the slow link."""
import json, os, sys, hashlib, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

URL = "http://127.0.0.1:8002/v1/embeddings"
INSTRUCT = "Instruct: Given a question about a past conversation, retrieve memory records that help answer it\nQuery: "
CACHE = os.path.expanduser("~/v3_emb/cache.npz")


def key(t):
    return hashlib.sha256(t.encode()).hexdigest()[:32]


def fetch(batch):
    body = json.dumps({"model": "qwen3-embedding-8b", "input": batch}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    data = json.loads(urllib.request.urlopen(req, timeout=600).read())["data"]
    v = np.array([d["embedding"] for d in sorted(data, key=lambda d: d["index"])], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def embed(texts, store):
    todo = [t for t in dict.fromkeys(texts) if key(t) not in store]
    batches = [todo[i:i + 32] for i in range(0, len(todo), 32)]
    with ThreadPoolExecutor(8) as ex:
        for b, v in zip(batches, ex.map(fetch, batches)):
            for t, x in zip(b, v):
                store[key(t)] = x
    return np.stack([store[key(t)] for t in texts])


def main():
    job = json.load(sys.stdin)
    store = dict(np.load(CACHE)) if os.path.exists(CACHE) else {}
    n0 = len(store)
    mats = {g: embed(d, store) for g, d in job["docs"].items()}
    qv = embed([INSTRUCT + q for g, q in job["queries"]], store)
    if len(store) > n0:                      # atomic: concurrent jobs never read a half-written cache
        tmp = f"{CACHE}.{os.getpid()}.tmp.npz"
        np.savez(tmp, **store)
        os.replace(tmp, CACHE)
    out = {}
    for (g, q), v in zip(job["queries"], qv):
        out[g + "\t" + q] = np.argsort(-(mats[g] @ v)).tolist()
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
