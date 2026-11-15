"""Candidate 2 on NewtonBench (ICLR 2026): does cross-task experience memory help a law-discovery agent?
Reuses NewtonBench's own vanilla agent loop and evaluate_law; model calls are rerouted to our vLLM servers and an
optional memory block is appended to the agent's system prompt. Split by domain: m0-m5 = experience (train),
m6-m11 = test (transfer to unseen domains). Systems: vanilla_equation + simple_system, noise 0.
usage (H20, in ~/data/simworld/ext/NewtonBench, alfenv or system python with numpy/scipy/sympy):
  python newton.py run --mods train --model Qwen3.5-9B --tag nb_train9b
  python newton.py run --mods test  --model Qwen3.5-9B --tag nb_C0
  python newton.py run --mods test  --model Qwen3.5-9B --tag nb_C3 --mem lessons --mem-src nb_lessons.json
  python newton.py curate --src nb_train9b --out nb_lessons.json
  python newton.py score TAG ...
env: NB_PORTS='Qwen3.5-9B=18081,18082;Qwen3.8-27B-FP8=18091,18092'  NB_JUDGE=Qwen3.8-27B-FP8"""
import argparse, collections, importlib, itertools, json, os, re, sys, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

NB = os.path.expanduser("~/data/simworld/ext/NewtonBench")
OUT = os.path.expanduser("~/data/rtm/c2/out")
sys.path.insert(0, NB)
PORTS = {k: v.split(",") for k, v in (x.split("=") for x in os.environ.get("NB_PORTS", "").split(";") if x)}
JUDGE = os.environ.get("NB_JUDGE", "Qwen3.8-27B-FP8")
_n = itertools.count()
THINK = os.environ.get("NB_THINK", "0") == "1"


def our_llm(messages, model_name, keys=None, temperature=0.4, trial_info=None):
    m = JUDGE if model_name == "judge" else model_name
    ports = PORTS[m]; port = ports[next(_n) % len(ports)]
    think = THINK if m != JUDGE or model_name != "judge" else False
    body = {"model": m, "messages": messages, "max_tokens": 6000 if think else 3000, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    err = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(req, timeout=900) as r: d = json.loads(r.read())
            c = d["choices"][0]["message"]; u = d.get("usage", {})
            return c.get("content") or "", "", u.get("total_tokens", 0)
        except Exception as e: err = e
    raise err


import types
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=None))   # NewtonBench's API client; unused (calls go to our_llm)
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *a, **k: None))
import utils.call_llm_api as cla
cla.call_llm_api = our_llm
import utils.vanilla_agent as va
va.call_llm_api = our_llm
import modules.common.evaluation as ev
ev.call_llm_api = our_llm
BASE = va.BASE_PROMPT

MODS = ["m0_gravity", "m1_coulomb_force", "m2_magnetic_force", "m3_fourier_law", "m4_snell_law", "m5_radioactive_decay",
        "m6_underdamped_harmonic", "m7_malus_law", "m8_sound_speed", "m9_hooke_law", "m10_be_distribution", "m11_heat_transfer"]
SYSTEMS = ["vanilla_equation", "simple_system"]


def tasks(which):
    mods = MODS[:6] if which == "train" else MODS[6:]
    out = []
    for m in mods:
        laws = importlib.import_module(f"modules.{m}.laws")
        for d in ("easy", "medium", "hard"):
            for v in laws.get_available_law_versions(d):
                for s in SYSTEMS: out.append((m, d, v, s))
    return out


_lock = threading.Lock()


def one(t, a, memtxt):
    m, d, v, s = t
    mod = importlib.import_module(f"modules.{m}")   # BASE_PROMPT (with memory) is set once per process in run()
    try:
        res = va.conduct_exploration(mod, a.model, 0.0, difficulty=d, system=s, law_version=v, max_turns=a.max_turns)
        law = res.get("submitted_law")
        evm = mod.evaluate_law(law or "", param_description=mod.PARAM_DESCRIPTION, difficulty=d, law_version=v,
                               judge_model_name="judge") if law else {"rmsle": float("nan"), "exact_accuracy": 0.0}
    except Exception as e:
        return {"mod": m, "diff": d, "ver": v, "sys": s, "error": repr(e)[:300], "exact": 0.0, "rmsle": None}
    hist = res.get("chat_history", [])
    traj = "\n".join(f"[{x['role']}] {x['content'][:1500]}" for x in hist[2:])
    return {"mod": m, "diff": d, "ver": v, "sys": s, "exact": float(evm.get("exact_accuracy", 0.0)),
            "rmsle": None if evm.get("rmsle") != evm.get("rmsle") else evm.get("rmsle"), "law": law,
            "rounds": res.get("rounds"), "tokens": res.get("total_tokens"), "task_prompt": hist[1]["content"][:3000] if len(hist) > 1 else "",
            "traj": traj[-8000:]}


def run(a):
    ts = tasks(a.mods)[: a.limit or None]
    path = os.path.join(OUT, f"{a.tag}.jsonl"); os.makedirs(OUT, exist_ok=True)
    key = lambda r: (r["mod"], r["diff"], r["ver"], r["sys"])
    done = {key(json.loads(l)) for l in open(path)} if os.path.exists(path) else set()
    ts = [t for t in ts if t not in done]
    memtxt = ""
    if a.mem == "lessons":
        L = json.load(open(os.path.join(OUT, a.mem_src)))
        memtxt = "\n\n**Lessons from your earlier law-discovery missions (in other domains):**\n" + "\n".join(f"- {x}" for x in L)
    elif a.mem == "raw":
        src = [json.loads(l) for l in open(os.path.join(OUT, f"{a.mem_src}.jsonl"))]
        good = sorted([r for r in src if r["exact"] > 0], key=lambda r: r["rounds"] or 99)[:2]
        memtxt = ("\n\n**Two of your earlier successful missions (other domains), for reference:**\n" +
                  "\n\n".join(f"Mission: {r['task_prompt'][:800]}\n{r['traj'][-3000:]}" for r in good))
    va.BASE_PROMPT = BASE + memtxt
    with open(path, "a") as f, ThreadPoolExecutor(a.par) as ex:
        for r in ex.map(lambda t: one(t, a, memtxt), ts):
            f.write(json.dumps(r) + "\n"); f.flush()
    score([a.tag])


CUR = ("Below are records of an AI agent trying to discover hidden physical laws (possibly altered from real physics) "
       "by running experiments, some successful and some failed. Write the most useful, GENERAL lessons for discovering "
       "laws in NEW, unseen domains: experiment-design strategies, data-analysis steps (e.g. log-log fits, testing "
       "exponents, checking functional forms), common failure modes and how to avoid them, how to verify before "
       "submitting. Do not mention specific laws or constants from these records. Return a JSON list of at most 20 short "
       "strings.\n\n{trajs}")


def curate(a):
    src = [json.loads(l) for l in open(os.path.join(OUT, f"{a.src}.jsonl"))]
    good = [r for r in src if r["exact"] > 0][:8]; bad = [r for r in src if r["exact"] == 0 and not r.get("error")][:8]
    trajs = "\n\n".join(f"### {'SUCCESS' if r in good else 'FAILURE'} ({r['mod']}, {r['diff']}, {r['sys']})\n"
                        f"{r['task_prompt'][:600]}\n{r['traj'][-2500:]}\nSubmitted: {r.get('law')}" for r in good + bad)
    global THINK; THINK = True
    txt, _, _ = our_llm([{"role": "user", "content": CUR.format(trajs=trajs)}], JUDGE, temperature=0.6)
    m = re.search(r"\[.*\]", txt, re.S); L = json.loads(m.group(0)) if m else []
    json.dump(L, open(os.path.join(OUT, a.out), "w"), indent=1); print(len(L), "lessons"); print("\n".join(L))


def score(tags):
    for tag in tags:
        rs = [json.loads(l) for l in open(os.path.join(OUT, f"{tag}.jsonl"))]
        by = collections.defaultdict(list)
        for r in rs: by[r["diff"]].append(r["exact"])
        print(f"{tag}: n={len(rs)} exact {100 * sum(r['exact'] for r in rs) / max(1, len(rs)):.1f}  errors {sum(1 for r in rs if r.get('error'))}  "
              + " ".join(f"{k}:{100 * sum(v) / len(v):.0f}" for k, v in sorted(by.items())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("tags", nargs="*")
    ap.add_argument("--mods", default="test"); ap.add_argument("--model", default="Qwen3.5-9B"); ap.add_argument("--tag", default="")
    ap.add_argument("--mem", default=""); ap.add_argument("--mem-src", default=""); ap.add_argument("--par", type=int, default=32)
    ap.add_argument("--max-turns", type=int, default=10); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--src", default=""); ap.add_argument("--out", default="")
    a = ap.parse_args()
    {"run": run, "curate": curate, "score": lambda a: score(a.tags)}[a.cmd](a)
