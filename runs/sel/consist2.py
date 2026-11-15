"""HF vs vLLM consistency on long BEAM inputs (10-31 audit): same model, same fmt strings.
usage (inside the vLLM image): python3 consist2.py MODELDIR GROUPS.jsonl PORT [--n 20]; exits 1 if they disagree."""
import json, sys, urllib.request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
md, gf, port = sys.argv[1:4]; n = 20
SFX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
tok = AutoTokenizer.from_pretrained(md); m = AutoModelForCausalLM.from_pretrained(md, torch_dtype=torch.bfloat16).cuda().eval()
y, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
sfx = tok(SFX, add_special_tokens=False)["input_ids"]
rows = [json.loads(l) for l in open(gf)][: n]
def rank(v): o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v); [r.__setitem__(i, k) for k, i in enumerate(o)]; return r
def spear(a, b):
    ra, rb = rank(a), rank(b); ma = sum(ra) / len(ra); mb = sum(rb) / len(rb)
    c = sum((x - ma) * (w - mb) for x, w in zip(ra, rb)); d = (sum((x - ma) ** 2 for x in ra) * sum((w - mb) ** 2 for w in rb)) ** .5
    return c / d
sp = []; md_p = []
for r in rows:
    q = tok(r["query"], add_special_tokens=False)["input_ids"]; hf = []
    for d in r["docs"]:
        body = d[: -len(SFX)] if d.endswith(SFX) else d
        ids = torch.tensor([q + tok(body, add_special_tokens=False)["input_ids"] + sfx]).cuda()
        with torch.no_grad(): lg = m(input_ids=ids).logits[0, -1].float()
        hf.append(torch.sigmoid(lg[y] - lg[no]).item())
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/rerank", data=json.dumps({"model": "sel", "query": r["query"], "documents": r["docs"]}).encode(), headers={"Content-Type": "application/json"})
    res = json.loads(urllib.request.urlopen(req, timeout=300).read())["results"]
    vl = [0.0] * len(hf)
    for x in res: vl[x["index"]] = x["relevance_score"]
    sp.append(spear(hf, vl)); md_p.append(max(abs(a - b) for a, b in zip(hf, vl)))
    if len(sp) == 1: print("example hf", [round(x, 4) for x in hf[:5]], "vllm", [round(x, 4) for x in vl[:5]])
print(f"spearman mean {sum(sp) / len(sp):.4f} min {min(sp):.4f}; max |dP| {max(md_p):.4f}")
sys.exit(0 if min(sp) > 0.98 and max(md_p) < 0.10 else 1)
