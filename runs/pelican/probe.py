"""Black-box check that astra is not luna behind the Plus proxy: random-number habits + exact multiplication.
One attempt per call, no retries; every reply is logged to probe.jsonl."""
import json, os, random, re, time, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

cfg = open(os.path.expanduser("~/.local/share/codex-workflow/llm-proxy-private/config.yaml")).read()
KEY = re.findall(r'^\s*-\s*"?([A-Za-z0-9_\-\.]{12,})"?\s*$', cfg[cfg.find("api-keys"):][:800], re.M)[0]
MODELS = ["gpt-6-astra", "gpt-6-luna"]
RNG = "Write 40 random integers between 1 and 100, separated by single spaces. Output only the numbers."
rnd = random.Random(7)
MUL = [(rnd.randrange(10**8, 10**9), rnd.randrange(10**8, 10**9)) for _ in range(10)]


def call(model, prompt, effort):
    body = {"model": model, "input": [{"role": "user", "content": prompt}], "reasoning": {"effort": effort}}
    req = urllib.request.Request("http://127.0.0.1:18319/v1/responses", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t = time.time()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read()[:200].decode(errors="replace")}
    except Exception as e:
        return {"error": str(e)[:200]}
    u = r.get("usage") or {}
    return {"served": r.get("model"), "seconds": round(time.time() - t, 1), "out": u.get("output_tokens"),
            "reasoning": (u.get("output_tokens_details") or {}).get("reasoning_tokens"),
            "text": "".join(c.get("text", "") for it in r.get("output", []) if it.get("type") == "message"
                            for c in it.get("content", []))}


jobs = [(m, "rng", i, RNG) for m in MODELS for i in range(5)] + \
       [(m, "mul", i, f"Compute {a} * {b}. Give only the exact integer result, no commas.") for m in MODELS for i, (a, b) in enumerate(MUL)]
with ThreadPoolExecutor(6) as ex:
    res = list(ex.map(lambda j: (j, call(j[0], j[3], "low")), jobs))
with open("probe.jsonl", "a") as f:
    for (m, kind, i, p), r in res:
        f.write(json.dumps({"time": time.strftime("%F %T"), "model": m, "kind": kind, "i": i, **r}) + "\n")

for m in MODELS:
    rs = [(j, r) for j, r in res if j[0] == m]
    errs = [r for j, r in rs if "error" in r]
    served = Counter(r.get("served") for j, r in rs if "error" not in r)
    nums = [int(x) for j, r in rs if j[1] == "rng" and "error" not in r for x in re.findall(r"\d+", r["text"])]
    ok = sum(1 for j, r in rs if j[1] == "mul" and "error" not in r
             and re.sub(r"\D", "", r["text"]) == str(MUL[j[2]][0] * MUL[j[2]][1]))
    mul_n = sum(1 for j, r in rs if j[1] == "mul" and "error" not in r)
    reas = sorted(r["reasoning"] or 0 for j, r in rs if j[1] == "mul" and "error" not in r)
    print(f"\n== {m}: served={dict(served)} errors={len(errs)}")
    print(f"  rng: {len(nums)} numbers, top10={Counter(nums).most_common(10)}")
    print(f"  rng first sample: {next((r['text'][:160] for j, r in rs if j[1]=='rng' and 'error' not in r), None)}")
    print(f"  mul exact: {ok}/{mul_n}; reasoning tokens (sorted)={reas}")
    if errs:
        print("  first error:", errs[0])
