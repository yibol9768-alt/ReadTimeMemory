"""Candidate 2 killer test on ALFWorld (text). ReAct agent (Yao et al. prompts, 2 same-type examples) with optional memory.
arms: none | raw (2 most similar successful train trajectories) | lessons (curated lesson list per task type)
usage (H20, alfenv):
  python alf.py run  --split train --n-per-type 50 --model Qwen3.5-9B --ports 18081,.. --tag train9b
  python alf.py run  --split valid_unseen --model Qwen3.5-9B --ports .. --tag C0_unseen [--mem raw|lessons --mem-src train9b]
  python alf.py curate --src train9b --model Qwen3.8-27B-FP8 --ports .. --out lessons_train9b.json
  python alf.py score  TAG ...
"""
import argparse, collections, glob, json, os, random, re, sys, threading, urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

D = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.expanduser(os.environ.get("ALFWORLD_DATA", "~/.cache/alfworld"))
PREFIXES = {"pick_and_place": "put", "pick_clean_then_place": "clean", "pick_heat_then_place": "heat",
            "pick_cool_then_place": "cool", "look_at_obj": "examine", "pick_two_obj": "puttwo"}
SYS = ("You are an agent acting in a text-based household environment (ALFWorld). Each turn, reply with exactly one line: "
       "either 'think: <your reasoning>' or one valid action (e.g. 'go to cabinet 1', 'take apple 1 from countertop 2', "
       "'open fridge 1', 'heat apple 1 with microwave 1', 'put apple 1 in/on countertop 1'). No other text.")


def post(port, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    err = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
        except Exception as e: err = e
    raise err


def chat(port, model, system, user, max_tokens=200, temp=0.0, think=False):
    r = post(port, {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "max_tokens": max_tokens, "temperature": temp, "chat_template_kwargs": {"enable_thinking": think}})
    u = r.get("usage", {})
    return (r["choices"][0]["message"]["content"] or ""), u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)


def games(split):
    fs = sorted(glob.glob(os.path.join(DATA, "json_2.1.1", split, "*", "trial_*", "game.tw-pddl")))
    return [f for f in fs if any(os.path.basename(os.path.dirname(os.path.dirname(f))).startswith(p) for p in PREFIXES)]


def gtype(f):
    d = os.path.basename(os.path.dirname(os.path.dirname(f)))
    return next(v for k, v in PREFIXES.items() if d.startswith(k))


def make_env(gamefile):
    import textworld, textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    eid = textworld.gym.register_games([gamefile], infos, max_episode_steps=60, wrappers=[AlfredDemangler(), AlfredInfos])
    return textworld.gym.make(eid)


def clean(a):
    a = a.strip().split("\n")[0].strip().lstrip(">").strip()
    return re.sub(r"^(action|act)\s*:\s*", "", a, flags=re.I)


def play(job):
    gf, a, mem = job
    t = gtype(gf); P = json.load(open(os.path.join(D, "alfworld_3prompts.json")))
    shots = "\n\n".join(P[f"react_{t}_{i}"] for i in (1, 0))
    env = make_env(gf); obs, info = env.reset()
    obs = "\n".join(obs.split("\n\n")[1:]) if "\n\n" in obs else obs
    task = obs.strip(); hist = f"{task}\n"; won, tok, steps = False, 0, 0
    memtxt = ""
    if mem:
        memtxt = ("\n\nNotes from your earlier experience in this environment (use them if helpful):\n" + mem(t, task)) if mem else ""
    for steps in range(1, a.max_steps + 1):
        user = (f"Here are two examples.\n\n{shots}{memtxt}\n\nHere is the task.\n{hist}>")
        act, n = chat(a.port_of(gf), a.model, SYS, user, max_tokens=a.act_tokens, think=a.think, temp=a.temp)
        tok += n; act = clean(act)
        if act.lower().startswith("think:"):
            obs = "OK."
        else:
            obs, score, done, info = env.step(act)
            won = bool(info.get("won")) if isinstance(info, dict) else False
            if done: hist += f"> {act}\n{obs}\n"; break
        hist += f"> {act}\n{obs}\n"
    env.close()
    return {"game": gf, "type": t, "won": won, "steps": steps, "tokens": tok, "traj": hist}


class Args:
    pass


def run(a):
    gs = games(a.split)
    if a.n_per_type:
        by = collections.defaultdict(list)
        for g in gs: by[gtype(g)].append(g)
        R = random.Random(20261030); gs = []
        for t in sorted(by): R.shuffle(by[t]); gs += by[t][: a.n_per_type]
    gs = gs[: a.limit or None]
    out = os.path.join(D, "out", f"{a.tag}.jsonl"); os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(l)["game"] for l in open(out)} if os.path.exists(out) else set()
    gs = [g for g in gs if g not in done]
    mem = None
    if a.mem == "lessons":
        L = json.load(open(os.path.join(D, a.mem_src)))
        mem = lambda t, task: "\n".join(f"- {x}" for x in L.get(t, []) + L.get("general", []))
    elif a.mem == "raw":
        src = [json.loads(l) for l in open(os.path.join(D, "out", f"{a.mem_src}.jsonl"))]
        good = [r for r in src if r["won"]]
        def mem(t, task):
            q = set(re.findall(r"[a-z]+", task.split("Your task is to:")[-1].lower()))
            cand = sorted([r for r in good if r["type"] == t],
                          key=lambda r: -len(q & set(re.findall(r"[a-z]+", r["traj"].split("Your task is to:")[-1].split("\n")[0].lower()))))
            return "\n\n".join(r["traj"][-3000:] for r in cand[:2])
    ports = a.ports.split(",")
    a.port_of = lambda g: ports[hash(g) % len(ports)]
    with open(out, "a") as f, ThreadPoolExecutor(a.par) as ex:
        for r in ex.map(play, [(g, a, mem) for g in gs]):
            f.write(json.dumps(r) + "\n"); f.flush()
    score([a.tag])


CUR = ("Below are trajectories of an agent in the ALFWorld household environment for tasks of type '{t}', some "
       "successful and some failed. Write the most useful, general lessons for solving NEW tasks of this type more "
       "reliably and in fewer steps: where objects usually are, required action sequences and their exact syntax, "
       "common mistakes and how to avoid them. Do not mention specific object numbers. Return a JSON list of at most 15 "
       "short strings.\n\n{trajs}")


def curate(a):
    src = [json.loads(l) for l in open(os.path.join(D, "out", f"{a.src}.jsonl"))]
    ports = a.ports.split(","); out = {}
    def one(t):
        rs = [r for r in src if r["type"] == t]
        pick = [r for r in rs if r["won"]][:6] + [r for r in rs if not r["won"]][:6]
        trajs = "\n\n".join(f"### {'SUCCESS' if r['won'] else 'FAILURE'}\n{r['traj'][-4000:]}" for r in pick)
        txt, _ = chat(ports[0], a.model, "You are a careful analyst.", CUR.format(t=t, trajs=trajs), max_tokens=12000, think=True)
        m = re.search(r"\[.*\]", txt, re.S)
        return t, (json.loads(m.group(0)) if m else [])
    with ThreadPoolExecutor(6) as ex: out = dict(ex.map(one, sorted(set(PREFIXES.values()))))
    json.dump(out, open(os.path.join(D, a.out), "w"), indent=1); print({k: len(v) for k, v in out.items()})


def score(tags):
    for tag in tags:
        rs = [json.loads(l) for l in open(os.path.join(D, "out", f"{tag}.jsonl"))]
        by = collections.defaultdict(list)
        for r in rs: by[r["type"]].append(r["won"])
        print(f"{tag}: n={len(rs)} success {100 * sum(r['won'] for r in rs) / max(1, len(rs)):.1f}  steps {sum(r['steps'] for r in rs) / max(1, len(rs)):.1f}  "
              + " ".join(f"{k}:{100 * sum(v) / len(v):.0f}" for k, v in sorted(by.items())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("tags", nargs="*")
    ap.add_argument("--split", default="valid_unseen"); ap.add_argument("--n-per-type", type=int, default=0)
    ap.add_argument("--model", default="Qwen3.5-9B"); ap.add_argument("--ports", default=""); ap.add_argument("--tag", default="")
    ap.add_argument("--mem", default=""); ap.add_argument("--mem-src", default=""); ap.add_argument("--par", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=50); ap.add_argument("--act-tokens", type=int, default=150)
    ap.add_argument("--think", type=int, default=0); ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--src", default=""); ap.add_argument("--out", default="")
    a = ap.parse_args(); a.think = bool(a.think)
    {"run": run, "curate": curate, "score": lambda a: score(a.tags)}[a.cmd](a)
