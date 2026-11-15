import json,sys,collections,random,os
P=sys.argv[1]; route=json.load(open(P+"route.json"))
def load(f):
    d={}
    for l in open(f):
        r=json.loads(l); d.setdefault(r["cond"],{})[r["qid"]]=r["score"]
    return d
fs=("judged.jsonl","judged_g2.jsonl","judged_g3.jsonl")
G=[load(P+f) for f in fs]; F=[load(P+"fair/"+f) for f in fs]
def a(src,c,q): return sum(g[c][q] for g in src)/3
qs=sorted(route)
new={q:(a(G,"note_ly~m0",q) if route[q]=="POINT" else a(G,"lychee_b60~m0",q)) for q in qs}
X={"ours_note":new,**{k:{q:a(F,k,q) for q in qs} for k in ("ours60","hs_cf","hs_m0")}}
print({k:round(100*sum(v.values())/len(qs),1) for k,v in X.items()})
conv=collections.defaultdict(list)
for q in qs: conv[q.split("-")[0]].append(q)
random.seed(0)
for o in ("ours60","hs_cf","hs_m0"):
    d={q:new[q]-X[o][q] for q in qs}; bs=[]
    for _ in range(5000):
        Q=[q for c in random.choices(list(conv),k=len(conv)) for q in conv[c]]; bs.append(100*sum(d[q] for q in Q)/len(Q))
    bs.sort(); print(f"ours_note - {o}: {100*sum(d.values())/len(qs):+.1f} [{bs[125]:+.1f},{bs[4874]:+.1f}]")
