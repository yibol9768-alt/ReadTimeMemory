"""Pointwise INNERSEL/full-inner inference only; common switches for yi/jia/A0/RR8/MR4.
No training code calls this module. Neighbours affect scoring input only; pack keeps original rounds.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from k1 import rounds_of, pack, _post
from fmt import doc_str, split_round, round_prev, query_str, SFX
from infer import OFF_PFX, OFF_INST, plain_doc
from inner_eval import B, validate_selection, inner_questions


def neighbours(R, i):
    return [(R[j][1], R[j][2]) for j in (i - 1, i + 1) if 0 <= j < len(R) and R[j][0] == R[i][0]]


def fuse_order(indices, scores, weight=0.0, k=60):
    import math
    if any(i not in scores or not math.isfinite(scores[i]) for i in indices): raise ValueError("missing/nonfinite ranking score")
    if not 0 <= weight <= 1: raise ValueError("RRF weight outside [0,1]")
    model = sorted(indices, key=lambda i: (-scores[i], indices.index(i)))
    if not weight: return model
    mr = {i: r for r, i in enumerate(model, 1)}; ret = {i: r for r, i in enumerate(indices, 1)}
    return sorted(indices, key=lambda i: (-((1 - weight) / (k + mr[i]) + weight / (k + ret[i])), ret[i]))


def rank_inner(tag, benches, sel=False, port=None, model="sel", style=None, score_key=None,
               neighbour_context=False, rrf_weight=0.0, smoke=False, *, par=None):
    """Score up to par questions concurrently (INNER_RANK_PAR, default 16)."""
    par = int(os.environ.get("INNER_RANK_PAR", "16")) if par is None else par
    if not isinstance(par, int) or par < 1: raise ValueError("par must be a positive integer")
    frozen = json.load(open("DATA_FREEZE_STAGE23.json"))["bench"]
    style = style or ("official" if score_key == "rr8" else "selector")
    if sel: frozen = validate_selection(frozen, json.load(open("INNERSEL_STAGE23.json")))
    if neighbour_context and not port: raise ValueError("neighbour context requires fresh model scores and --port")
    for b in benches:
        qs = inner_questions(b, frozen, sel, smoke)
        def score_question(q):
            R = rounds_of(q); prev = round_prev(q)
            pool = json.load(open(f"pool_s23/{q['question_id']}.json")); P = pool["fused"][:600]
            if not P or len(P) != len(set(P)): raise ValueError("empty/duplicate retrieval candidates")
            import math
            cached = pool.get(score_key, {}) if score_key else {}
            complete = all(str(i) in cached and isinstance(cached[str(i)], (int, float))
                           and math.isfinite(cached[str(i)]) for i in P)
            same_input = score_key != "rr8" or style == "official"
            if score_key and not neighbour_context and complete and same_input:
                scores = {i: cached[str(i)] for i in P}
            else:
                if score_key and not neighbour_context and (not complete or not same_input) and not port:
                    raise ValueError("cached scores do not cover top 600 or requested input; fresh model scoring requires --port")
                if not port: raise ValueError("fresh model scoring requires --port")
                docs = []
                for i in P:
                    adj = neighbours(R, i)
                    if style == "selector":
                        d = doc_str(R[i][1], *split_round(R[i][2]), prev=prev[i],
                                    neighbour_context=neighbour_context, neighbours=adj)
                    else:
                        d = (f"<Document>: ({R[i][1]}) {R[i][2][:3000]}" + SFX if style == "official" else
                             plain_doc(R[i][1], *split_round(R[i][2]), prev=prev[i]))
                        if neighbour_context:
                            ctx = "\n[Adjacent rounds]\n" + "\n".join(f"({dt}) {t[:700]}" for dt, t in adj)
                            d = d[:-len(SFX)] + ctx + SFX if style == "official" else d + ctx
                    docs.append(d)
                query = (query_str(q["question"], q["question_date"], []) if style == "selector" else
                         OFF_PFX + "<Instruct>: " + OFF_INST + "\n<Query>: " + q["question"] + "\n" if style == "official" else q["question"])
                scores = {}
                for start in range(0, len(P), 64):
                    batch = P[start:start + 64]
                    result = _post(f"http://127.0.0.1:{port}/v1/rerank", {"model": model, "query": query, "documents": docs[start:start + 64]})
                    import math
                    seen = set()
                    for item in result["results"]:
                        j, v = item["index"], item["relevance_score"]
                        if not isinstance(j, int) or not 0 <= j < len(batch) or j in seen or not math.isfinite(v):
                            raise ValueError("invalid rerank result")
                        seen.add(j); scores[batch[j]] = v
                    if len(seen) != len(batch): raise ValueError("incomplete rerank result")
            order = fuse_order(P, scores, rrf_weight)
            order += [i for i in pool["fused"] if i not in set(P)]
            order += [i for i in range(len(R)) if i not in set(order)]
            return [json.dumps({**pack(q, R, order, budget), "rank": order[:len(P)],
                                "neighbour_context": neighbour_context, "rrf_weight": rrf_weight}) + "\n"
                    for budget in B]

        with open(f"ctx_{tag}_inner_{b}.jsonl", "w") as out:
            if par == 1:
                for result in map(score_question, qs):
                    out.writelines(result)
            else:
                with ThreadPoolExecutor(max_workers=par) as executor:
                    # map yields results in input order, regardless of completion order.
                    for result in executor.map(score_question, qs):
                        out.writelines(result)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tag"); ap.add_argument("benches"); ap.add_argument("--sel", action="store_true")
    ap.add_argument("--port"); ap.add_argument("--model", default="sel")
    ap.add_argument("--style", choices=["selector", "official", "plain"],
                    help="default: official for --score-key rr8, selector otherwise")
    ap.add_argument("--choice", help="yi choice JSON: apply its frozen inference switches to any model/baseline")
    ap.add_argument("--score-key"); ap.add_argument("--neighbour-context", action="store_true")
    ap.add_argument("--rrf-weight", type=float, default=0.0)
    ap.add_argument("--smoke", action="store_true", help="score the fixed 10-question/benchmark INNERSEL subset; requires --sel")
    a = ap.parse_args()
    if a.choice:
        opt = json.load(open(a.choice))["inference"]["options"]
        a.neighbour_context, a.rrf_weight = opt["neighbour_context"], opt["rrf_weight"]
    rank_inner(a.tag, a.benches.split(","), a.sel, a.port, a.model, a.style,
                                    a.score_key, a.neighbour_context, a.rrf_weight, a.smoke)
