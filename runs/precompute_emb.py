import json, sys, time
sys.argv = ["x"]
from mem import retrieve as R
texts = []
for ci in range(10):
    texts += [r["text"] for r in json.load(open(f"runs/locomo/mem_{ci}.json"))["rec_sess"]]
t = time.time(); R.embed(texts); print("records embedded", len(texts), round(time.time() - t), "s", flush=True)
