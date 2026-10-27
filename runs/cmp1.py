import json,sys,collections,random
def load(f,conds):
    d={}
    for l in open(f):
        r=json.loads(l)
        if r["cond"] in conds: d.setdefault(r["cond"],{})[r["qid"]]=r
    return d
P="runs/beam1m/pro6/"
A=load(P+"judged.jsonl",{"note_ly~m0","note2_hs~m0","lychee_cf~m0","lychee_b60~m0"})
B=load(P+"fair/judged.jsonl",{"ours60","hs_m0","hs_cf"})
A.update(B)
route=json.load(open(P+"route.json"))
qs=sorted(A["ours60"])
ab=collections.defaultdict(list)
for q in qs: ab[A["ours60"][q]["ability"]].append(q)
names=["note_ly~m0","ours60","hs_m0","hs_cf","note2_hs~m0"]
m=lambda c,Q: 100*sum(A[c][q]["score"] for q in Q)/len(Q)
print("%-22s"%"",*["%11s"%n for n in names])
print("%-22s"%"ALL",*["%11.1f"%m(c,qs) for c in names])
for R in("POINT","AREA"):
    Q=[q for q in qs if route.get(q)==R]; print("%-22s"%(R+str(len(Q))),*["%11.1f"%m(c,Q) for c in names])
for a,Q in sorted(ab.items()): print("%-22s"%a,*["%11.1f"%m(c,Q) for c in names])
print("ctx chars", {c:int(sum(A[c][q].get("ctx_chars",0) for q in qs)/len(qs)) for c in names})
for o in names[1:]:
    w=sum(A["note_ly~m0"][q]["score"]>A[o][q]["score"] for q in qs); l=sum(A["note_ly~m0"][q]["score"]<A[o][q]["score"] for q in qs)
    print("note_ly vs",o,"W/L",w,l)
