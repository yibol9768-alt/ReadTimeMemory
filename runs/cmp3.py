import json,sys,collections,random,os
P=sys.argv[1]; route=json.load(open(P+"route.json"))
def load(f):
    d={}
    if not os.path.exists(f): return d
    for l in open(f):
        r=json.loads(l); d.setdefault(r["cond"],{})[r["qid"]]=r["score"]
    return d
G=[load(P+f) for f in ("judged.jsonl","judged_g2.jsonl","judged_g3.jsonl")]
F=[load(P+"fair/"+f) for f in ("judged.jsonl","judged_g2.jsonl","judged_g3.jsonl")]
def avg(src,c,qs):
    out={}
    for q in qs:
        v=[g[c][q] for g in src if c in g and q in g[c]]
        if len(v)==3: out[q]=sum(v)/3
    return out
pts=[q for q,r in route.items() if r=="POINT"]
X={"note_ly":avg(G,"note_ly~m0",pts),"lychee_cf":avg(G,"lychee_cf~m0",pts),"hs_cf":avg(F,"hs_cf",pts),"hs_m0":avg(F,"hs_m0",pts)}
qs=[q for q in pts if all(q in v for v in X.values())]
print("n",len(qs),{k:round(100*sum(v[q] for q in qs)/len(qs),1) for k,v in X.items()})
conv=collections.defaultdict(list)
for q in qs: conv[q.split("-")[0]].append(q)
random.seed(0)
for o in ("lychee_cf","hs_cf","hs_m0"):
    d={q:X["note_ly"][q]-X[o][q] for q in qs}; m=100*sum(d.values())/len(qs)
    bs=[]
    for _ in range(5000):
        s=[c for c in random.choices(list(conv),k=len(conv))]; Q=[q for c in s for q in conv[c]]
        bs.append(100*sum(d[q] for q in Q)/len(Q))
    bs.sort(); w=sum(v>0 for v in d.values()); l=sum(v<0 for v in d.values())
    print(f"note_ly - {o}: {m:+.1f} [{bs[125]:+.1f},{bs[4874]:+.1f}] W/L {w}:{l}")
ab=collections.defaultdict(list)
J=load(P+"judged.jsonl")
abil={}
for l in open(P+"judged.jsonl"):
    r=json.loads(l); abil[r["qid"]]=r["ability"]
for q in qs: ab[abil[q]].append(q)
for a,Q in sorted(ab.items()): print(f"{a:26s}{len(Q):3d}",*[f"{k}={100*sum(X[k][q] for q in Q)/len(Q):.1f}" for k in X])
