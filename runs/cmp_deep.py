import json,collections,random,sys
fs=("judged.jsonl","judged_g2.jsonl","judged_g3.jsonl")
def load(f):
    d={}
    for l in open(f):
        try: r=json.loads(l)
        except Exception: continue
        d.setdefault(r["cond"],{})[r["qid"]]=r["score"]
    return d
def boot(d,qs):
    conv=collections.defaultdict(list)
    for q in qs: conv[q.split("-")[0]].append(q)
    random.seed(0); bs=[]
    for _ in range(5000):
        Q=[q for c in random.choices(list(conv),k=len(conv)) for q in conv[c]]; bs.append(100*sum(d[q] for q in Q)/len(Q))
    bs.sort(); return f"{100*sum(d.values())/len(qs):+.1f} [{bs[125]:+.1f},{bs[4874]:+.1f}]"
for P in sys.argv[1:]:
    route=json.load(open(P+"route.json")); G=[load(P+f) for f in fs]; F=[load(P+"fair/"+f) for f in fs]
    qs=sorted(route); pts=[q for q in qs if route[q]=="POINT"]; ar=[q for q in qs if route[q]=="AREA"]
    a=lambda S,c,q: sum(g[c][q] for g in S)/3
    D={q:a(G,"note3_deep~m0",q) for q in qs}
    R={q:(a(G,"note3_ly~m0",q) if route[q]=="POINT" else a(G,"lychee_b60~m0",q)) for q in qs}
    Y={k:{q:a(F,k,q) for q in qs} for k in ("ours60","hs_cf","hs_m0")}
    m=lambda x,Q: round(100*sum(x[q] for q in Q)/len(Q),1)
    print(P,"ALL deep",m(D,qs),"routed",m(R,qs),{k:m(v,qs) for k,v in Y.items()})
    print("  POINT deep",m(D,pts),"routed",m(R,pts),"| AREA deep",m(D,ar),"b60",m(R,ar),"hs_m0",m(Y["hs_m0"],ar))
    print("  deep-routed",boot({q:D[q]-R[q] for q in qs},qs),"| AREA deep-b60",boot({q:D[q]-R[q] for q in ar},ar))
    for k in Y: print("  deep-"+k,boot({q:D[q]-Y[k][q] for q in qs},qs))
