import json,sys,collections,random
P=sys.argv[1]; route=json.load(open(P+"route.json")); d={};ab={}
for l in open(P+"judged.jsonl"):
    try: r=json.loads(l)
    except: continue
    d.setdefault(r["cond"],{})[r["qid"]]=r["score"]; ab[r["qid"]]=r["ability"]
qs=sorted(route); pts=[q for q in qs if route[q]=="POINT"]; ar=[q for q in qs if route[q]=="AREA"]
print({c:len(v) for c,v in d.items()})
D=d["note3_deep~m0"]; H=d["hindsight~m0"]; HC={q:(d["cf_hindsight~m0"][q] if route[q]=="POINT" else H[q]) for q in qs}
m=lambda x,Q: round(100*sum(x[q] for q in Q)/len(Q),1)
for n,Q in (("ALL",qs),("POINT",pts),("AREA",ar)): print(n,len(Q),"ours",m(D,Q),"hs_cf",m(HC,Q),"hs",m(H,Q))
conv=collections.defaultdict(list)
for q in qs: conv[q.split("-")[0]].append(q)
random.seed(0)
for nm,X in (("hs_cf",HC),("hs",H)):
    dd={q:D[q]-X[q] for q in qs}; bs=[]
    for _ in range(5000):
        Q=[q for c in random.choices(list(conv),k=len(conv)) for q in conv[c]]; bs.append(100*sum(dd[q] for q in Q)/len(Q))
    bs.sort(); print(f"ours-{nm} {100*sum(dd.values())/len(qs):+.1f} [{bs[125]:+.1f},{bs[4874]:+.1f}] W/L {sum(v>0 for v in dd.values())}:{sum(v<0 for v in dd.values())}")
g=collections.defaultdict(list)
for q in qs: g[ab[q]].append(q)
for a,Q in sorted(g.items()): print(f"  {a:26s}{len(Q):3d}",m(D,Q),m(HC,Q))
