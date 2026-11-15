"""LongMemEval-V2 Small runner (our own minimal harness; official system prompts, message layout and scoring functions).
  answer: python3 runs/lmev2/run_lme.py answer --method none|bm25|uistore|hybrid --tag T [--types abs,...] [--budget 60000]
  judge : python3 runs/lmev2/run_lme.py judge --tag T --method M        (LLM-judged types via Qwen, official prompts)
  report: python3 runs/lmev2/run_lme.py report
Reader = llm.MODEL (V3_MODEL env; DeepSeek for dev, Qwen3.5-9B for the official protocol). Question screenshots are
not sent (text-only); image questions are flagged in the output so they can be reported separately.
"""
import argparse, json, os, re, sys, collections
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "runs/lmev2/repo/evaluation"))
from mem import llm
from mem.uistore import UIIndex
import qa_eval_metrics as M

D = os.path.join(ROOT, "runs/lmev2")
SYS = {
    "web": ("You are an experienced colleague in a web browsing environment that has a customized magento-based shopping "
            "website, a customized magento-based shopping admin cms website, as well as a customized forum website based "
            "on reddit/postmill. Answer based on your memory of the environment. If you do not know the answer, output "
            "exactly \\boxed{UNKNOWN}. Do not guess. Never attempt to guess an answer if you are not sure. If you believe "
            "the question's construction/premise is wrong, provide an explanation in \boxed{} explaining why the question "
            "is flawed."),
    "enterprise": ("You are an experienced colleague working in a customized ServiceNow environment. Answer based on your "
                   "memory of the environment. If you do not know the answer, output exactly \\boxed{UNKNOWN}. Do not "
                   "guess. Never attempt to guess an answer if you are not sure. If you believe the question's "
                   "construction/premise is wrong, provide an explanation in \boxed{} explaining why the question is "
                   "flawed."),
}
_bm = {}


def bm25_ctx(domain, q, budget):
    with _lock:
        if domain not in _bm:
            _load_bm(domain)
    return _bm[domain].query(q, budget)


def _load_bm(domain):
    if domain not in _bm:
        import importlib.util
        spec = importlib.util.spec_from_file_location("rp", os.path.join(D, "bm25lib.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        _bm[domain] = m.BM25(domain)


_ui = {}
import threading
_lock = threading.Lock()


def context(method, q, budget):
    dom = q["domain"]
    if method == "none":
        return ""
    if method == "bm25":
        return bm25_ctx(dom, q["question"], budget)
    with _lock:
        if dom not in _ui:
            _ui[dom] = UIIndex(os.path.join(D, f"store_small_{dom}"))
    if method == "uistore":
        return _ui[dom].query(q["question"], budget)[0]
    if method in ("replay", "replay_bm25", "replay_cov", "replay_names", "replay2_bm25", "replay2_facts"):
        from mem.replay import ReplayIndex
        v2 = method.startswith("replay2")
        key = ("rp2_" if v2 else "rp_") + dom
        with _lock:
            if key not in _ui:
                _ui[key] = ReplayIndex(os.path.join(D, f"replay{'2' if v2 else ''}_small_{dom}"))
                _ui[key].absent([])   # build the lookup text once, before threads share it
        if method == "replay":
            return _ui[key].query(q["question"], budget)[0]
        return _ui[key].query(q["question"], budget // 2, coverage={"replay_cov": True, "replay_names": "names", "replay2_facts": "facts"}.get(method, False))[0] + \
            "\n\n### Raw observation slices\n" + bm25_ctx(dom, q["question"], budget // 2)
    if method == "hybrid":   # half closed-world store, half raw BM25 slices
        return _ui[dom].query(q["question"], budget // 2)[0] + "\n\n### Raw observation slices\n" + \
            bm25_ctx(dom, q["question"], budget // 2)
    raise ValueError(method)


def out_path(tag, method):
    return os.path.join(D, "out", f"{tag}_{method}.jsonl")


def answer(a):
    qs = [json.loads(l) for l in open(os.path.join(D, "data/questions.jsonl"))]
    if a.types:
        qs = [q for q in qs if any(t == q["question_type"] or (t == "abs" and q["question_type"].endswith("-abs"))
                                   for t in a.types.split(","))]
    if a.ids_file:
        keep = set(open(a.ids_file).read().split())
        qs = [q for q in qs if q["id"] in keep]
    if a.limit:
        qs = qs[: a.limit]
    os.makedirs(os.path.join(D, "out"), exist_ok=True)
    p = out_path(a.tag, a.method)
    done = {json.loads(l)["id"] for l in open(p)} if os.path.exists(p) else set()
    todo = [q for q in qs if q["id"] not in done]

    def one(q):
        ctx = context(a.method, q, a.budget)
        user = "### Memory context:\n" + (ctx or "(empty)") + "\n\n### Question to answer:\n" + q["question"]
        resp = llm.call(SYS[q["domain"]], user, effort=a.effort, max_tokens=8000, tag="lme_" + a.method)
        parsed = M.extract_boxed_answer(resp or "")
        return {"id": q["id"], "domain": q["domain"], "type": q["question_type"], "image": bool(q.get("image")),
                "eval_function": q["eval_function"], "answer": q["answer"], "question": q["question"],
                "response": resp, "parsed": parsed, "unknown": M.is_unknown(parsed), "ctx_chars": len(ctx)}

    with ThreadPoolExecutor(a.par) as ex, open(p, "a") as f:
        for r in ex.map(one, todo):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
    print("answered", a.tag, a.method, len(todo))


def judge_llm(r, judge):
    kw = dict(question_text=r["question"], reference_answer=r["answer"], model_full_response=r["response"],
              model_final_answer=r["parsed"])
    msgs = (M._build_abstention_judge_messages if r["eval_function"].startswith("llm_abstention")
            else M._build_gotchas_judge_messages)(**kw)
    for i in range(4):
        txt = llm.call(msgs[0]["content"], msgs[1]["content"] + " " * i, effort="low", max_tokens=2000,
                       tag="lme_judge", model=judge)
        try:
            return M._parse_llm_binary_judgement(txt)[0]
        except Exception:
            continue
    return 0


def score(r, judge=None):
    if r["unknown"]:
        return 0
    name = M.eval_name(r["eval_function"])
    if name.startswith("llm_"):
        return r.get("llm_score") if judge is None else judge_llm(r, judge)
    return int(M.score_to_bool(M.eval_from_spec(r["eval_function"], r["parsed"], r["answer"])))


def judge(a):
    p = out_path(a.tag, a.method)
    rows = [json.loads(l) for l in open(p)]
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda r: score(r, a.judge) if M.eval_name(r["eval_function"]).startswith("llm_") else None, rows))
    for r, s in zip(rows, res):
        if s is not None:
            r["llm_score"] = s
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("judged", a.tag, a.method)


def report(a):
    import glob
    for p in sorted(glob.glob(os.path.join(D, "out", "*.jsonl"))):
        rows = [json.loads(l) for l in open(p)]
        by = collections.defaultdict(list)
        for r in rows:
            s = score(r)
            by[r["type"].replace("static-environment", "static").replace("dynamic-environment", "dynamic")].append(s)
        cells = {t: (f"{sum(x for x in v if x)}/{len(v)}" + ("" if all(x is not None for x in v) else "*"))
                 for t, v in sorted(by.items())}
        unk = sum(r["unknown"] for r in rows)
        print(os.path.basename(p), cells, f"UNKNOWN {unk}/{len(rows)}",
              f"ctx {sum(r['ctx_chars'] for r in rows) // max(1, len(rows))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["answer", "judge", "report"])
    ap.add_argument("--ids-file")
    ap.add_argument("--method", default="uistore"); ap.add_argument("--tag", default="dev")
    ap.add_argument("--types", default=""); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget", type=int, default=60000); ap.add_argument("--par", type=int, default=2)
    ap.add_argument("--judge", default="Qwen3.8-27B-FP8")
    ap.add_argument("--effort", default="medium")
    a = ap.parse_args()
    {"answer": answer, "judge": judge, "report": report}[a.step](a)
