"""Stage 2/3 v4.1b. All-data: `s23.py grid --all-data --arm yi`.
Legacy fold plan/run remains available. See S23_V41B.md for recipes and commands.
"""
import argparse, hashlib, json, math, os, random, subprocess, sys, tempfile, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
BENCH = ["beam", "lme", "loc", "ev", "ep"]
DEVSET = {"beam": ("beam_ex.json", "pool", "H"), "lme": ("q_lmes150.json", "pool_lmes150", "lmes150"), "loc": ("q_loc150.json", "pool_loc150", "loc150"),
          "ev": ("q_ev150.json", "pool_ev150", "ev150"), "ep": ("q_ep150.json", "pool_ep150", "ep150")}
IMG = "docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909"; MODELS = "/home/centuser/data/models"; A0 = "/models/qwen/Qwen3-Reranker-0.6B"
B5 = "1000,2000,4000,8000,16000"; PER_STAGE = 4000; TRAIN_COMMON = ["--lr", "1e-5", "--epochs", "1", "--max-d", "1100", "--w-bce", "0.2", "--w-teacher", "1.0", "--accum", "4"]
H = os.path.dirname(os.path.abspath(__file__))
PROTOCOL = {"score_depth": 600, "teacher_probability_clip": 1e-6, "preflight": 1}
_QUESTION_IDS_CACHE = {}


def _question_ids(path, *, unique=False):
    """Cache only identities, without retaining multi-GB question documents."""
    absolute = os.path.abspath(path)
    with open(path) as f:
        try:
            st = os.fstat(f.fileno())
        except (AttributeError, OSError):
            # File-like inputs without filesystem metadata cannot be memoized.
            return ({q["question_id"] for q in json.load(f)} if unique else
                    [q["question_id"] for q in json.load(f)])
        key = (absolute, st.st_size, st.st_mtime_ns)
        if key not in _QUESTION_IDS_CACHE:
            questions = json.load(f)
            # Preserve set-comprehension error order on malformed source identities.
            if unique: {q["question_id"] for q in questions}
            ids = [q["question_id"] for q in questions]
            for old in list(_QUESTION_IDS_CACHE):
                if old[0] == absolute: del _QUESTION_IDS_CACHE[old]
            _QUESTION_IDS_CACHE[key] = ids
        ids = _QUESTION_IDS_CACHE[key]
        return set(ids) if unique else ids


def preflight(a, stage1=False):
    from preflight_s23 import preflight as check
    return check(a, H, MODELS, stage1)


def sh(cmd, log=None):
    print("+", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    r = subprocess.run(cmd, shell=isinstance(cmd, str), stdout=open(log, "a") if log else None, stderr=subprocess.STDOUT if log else None)
    if r.returncode: sys.exit(f"FAIL ({r.returncode}): {cmd}")


def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
def lines(p): return [l for l in open(p)]


def all_namespace(a):
    teacher = getattr(a, "teacher", "rr8").replace("+", "_")
    config = getattr(a, "teacher_config", "")
    return f"s23/all_{teacher}" + ("_" + md5(config)[:12] if config else "")


def wd(a):
    if getattr(a, "all_data", False):
        root = grid_root(a) + "/plans" if getattr(a, "smoke", False) else all_namespace(a)
        return f"{root}/{a.arm}_s{a.seed}_g2{a.s2_groups}_r{a.r}_g3{a.s3_groups}"
    return f"s23/{a.arm}_{a.fold}_s{a.seed}_r{a.r}"


def _atomic_write(path, text):
    """Write via a temp file in the same directory, then os.replace (concurrent readers never see partial files)."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".tmp_")
    with os.fdopen(fd, "w") as f: f.write(text)
    os.replace(tmp, path)


def flops(path, world, accum, seed, epochs=1, w_base=0.0):
    """planned-path FLOPs (plan_flops.py inside the image, CPU only), cached by file md5 + parameters"""
    key = f"{md5(path)}_{world}_{accum}_{seed}_{epochs}_{w_base}"; cp = f"s23/cache/flops_{key}.json"
    if not os.path.exists(cp):
        out = subprocess.run(["docker", "run", "--rm", "--entrypoint", "python3", "-e", "CUDA_VISIBLE_DEVICES=", "-v", f"{H}:/w", "-v", f"{MODELS}:/models:ro", "-w", "/w", IMG,
                              "plan_flops.py", "--model", A0, "--data", "/w/" + os.path.relpath(path, H), "--epochs", str(epochs), "--max-d", "1100",
                              "--world", str(world), "--accum", str(accum), "--seed", str(seed), "--w-base", str(w_base), "--params", "595776512"],
                             capture_output=True, text=True)
        if out.returncode: sys.exit("FAIL plan_flops: " + out.stdout[-500:] + out.stderr[-500:])
        os.makedirs("s23/cache", exist_ok=True); _atomic_write(cp, out.stdout.strip().splitlines()[-1])
    return json.load(open(cp))["planned_flops"]


def source_parameters(kind, b, seed, teacher="rr8", teacher_config="", all_data=False):
    if kind not in ("teacher", "gold"): raise ValueError(f"unknown source kind: {kind}")
    flags = (["--teacher-only", "--n-groups", "20000"] if kind == "teacher" else
             ["--gold-rounds", "--no-set", "--hard-skip", "10", "--steps", "8"])
    extra = {"teacher": teacher, "teacher_config": teacher_config, "all_data": True} if all_data else {}
    if all_data and kind == "teacher": flags += ["--teacher", teacher, "--teacher-config", teacher_config]
    return {**extra, "kind": kind, "benchmark": b, "seed": seed, "qfile": f"q_train_{b}.json", "pooldir": "pool_s23",
            "flags": flags + ["--seed", str(seed)], "FMT_INST": os.environ.get("FMT_INST", "1")}


def source_inputs(params):
    # Defaults and formatting affect the bytes just as the command-line flags do.
    paths = [params["qfile"], "POOL_S23_MD5.json", "mkdata.py", "fmt.py", "k1.py", "d4data.py", "teacher_s23.py"]
    if params.get("all_data"):
        paths += ["DATA_FREEZE_STAGE23.json"]
        config = params.get("teacher_config")
        if config:
            paths.append(config)
            cfg = json.load(open(config))
            for name in params["teacher"].split("+"):
                entry = cfg.get(name, {})
                if "file" in entry:
                    paths += [entry["file"].replace("{qid}", q["question_id"]) for q in json.load(open(params["qfile"]))]
    return {p: md5(p) for p in paths}


def validate_source(path, params):
    rows = [json.loads(l) for l in lines(path)]
    if params.get("all_data"):
        allowed = _question_ids(params["qfile"], unique=True)
        if any(r.get("qid") not in allowed for r in rows): raise ValueError("source contains non-training qid")
    if not rows: raise ValueError(f"empty source: {path}")
    teacher = params["kind"] == "teacher"
    if teacher and len(rows) != 20000: raise ValueError(f"incomplete teacher source: {len(rows)} != 20000")
    for r in rows:
        if not isinstance(r.get("qid"), str) or not isinstance(r.get("query"), str):
            raise ValueError(f"invalid source group: {path}")
        docs, scores = r.get("docs", []), r.get("teacher", [])
        if not 5 <= len(docs) <= 16 or len(scores) != len(docs) or not all(isinstance(d, str) for d in docs):
            raise ValueError(f"invalid source candidates: {path}")
        if teacher:
            from teacher_s23 import logits
            logits(scores, r.get("teacher_format", "probability"))
            if params.get("all_data") and (r.get("teacher_format") != "logit" or r.get("teacher_name") != params["teacher"]):
                raise ValueError("source teacher metadata mismatch")
            if r.get("nolabel") is not True or len(docs) != 16 or any(t is None for t in scores):
                raise ValueError(f"invalid teacher-only source: {path}")
        elif r.get("nolabel") or r.get("nsel") != 0 or r.get("label") != 0:
            raise ValueError(f"invalid gold source parameters: {path}")
    return len(rows)


def source(kind, b, seed, teacher="rr8", teacher_config="", all_data=False):
    """Publish validated sources atomically; unmanifested/changed caches are regenerated.

    Data is replaced before its manifest. A crash between replacements leaves a cache miss,
    never an accepted partial source. Pool hashes are checked even on cache hits.
    """
    os.makedirs("s23/src", exist_ok=True); p = f"s23/src/{kind}_{b}_s{seed}.jsonl"
    params = source_parameters(kind, b, seed, teacher, teacher_config, all_data)
    if all_data:
        validate_train_files()
        token = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        p = f"s23/src/{kind}_{b}_s{seed}_{token}.jsonl"
    if os.path.exists(p) and os.path.getsize(p) == 0: raise ValueError(f"empty source: {p}")
    sh(["python3", "verify_pools.py", params["qfile"]])
    expected = {"version": 1, "parameters": params, "input_md5": source_inputs(params)}
    manifest = p + ".manifest.json"
    if os.path.exists(p) and os.path.exists(manifest):
        try:
            m = json.load(open(manifest))
            if (all(m.get(k) == v for k, v in expected.items()) and m.get("source_md5") == md5(p)
                    and m.get("line_count") == validate_source(p, params)):
                return p
        except (ValueError, KeyError, TypeError):
            pass  # corrupt/truncated cache: rebuild, rather than cycling its partial contents
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(p) + ".", suffix=".tmp", dir="s23/src"); os.close(fd)
    try:
        sh(["python3", "mkdata.py", params["qfile"], params["pooldir"], tmp] + params["flags"], log=p + ".log")
        count = validate_source(tmp, params)
        sh(["python3", "verify_pools.py", params["qfile"]])
        if source_inputs(params) != expected["input_md5"]: raise ValueError("source inputs changed during generation")
        m = {**expected, "source_md5": md5(tmp), "line_count": count}
        with open(tmp + ".manifest", "w") as f: json.dump(m, f, indent=1)
        os.replace(tmp, p)
        os.replace(tmp + ".manifest", manifest)
    finally:
        for f in (tmp, tmp + ".manifest"):
            if os.path.exists(f): os.remove(f)
    return p


def take(path, n, start=0, seed=0):
    """n rows starting at `start` of the seeded permutation of `path`; cycles (with a fresh seeded permutation) if needed"""
    if n < 0 or start < 0: raise ValueError("negative group count/offset")
    return cycle_rows(lines(path), start + n, seed, md5(path))[start:]


def cycle_rows(rows, n, seed, salt):
    if not rows: raise ValueError("empty source/remainder")
    out = []; k = 0
    while len(out) < n:
        idx = list(range(len(rows))); random.Random(f"{seed}-{k}-{salt}").shuffle(idx)
        out.extend(rows[i] for i in idx[:n - len(out)]); k += 1
    return out


def group_identity(row):
    """Content identity, independent of JSON formatting and candidate order (train shuffles docs)."""
    r = json.loads(row) if isinstance(row, str) else row
    key = [r["qid"], r["query"], sorted(r["docs"])]
    return hashlib.sha256(json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def assert_teacher_disjoint(s2, s3):
    ids = lambda rows: {group_identity(r) for r in rows if json.loads(r).get("nolabel")}
    overlap = ids(s2) & ids(s3)
    if overlap: raise ValueError(f"stage-2/stage-3 teacher identity overlap: {len(overlap)}")


def exposure_counts(s2, s3):
    c2, c3 = (Counter(map(group_identity, rows)) for rows in (s2, s3))
    return {i: {"stage2": c2[i], "stage3": c3[i], "repeat_count": max(0, c2[i] + c3[i] - 1)}
            for i in sorted(c2.keys() | c3.keys())}


def teacher_blocks(path, n2, n3, seed):
    # Reserve the FIRST seeded block before stage 2, including any compute top-up.
    ordered = take(path, len(lines(path)), seed=seed)
    if n3 > len(ordered): raise ValueError("stage-3 reservation exceeds teacher source")
    reserved, remainder = ordered[:n3], ordered[n3:]
    if not remainder: raise ValueError("empty source/remainder")
    # Preserve the seeded source order, including on subsequent passes (v3.4 section 5).
    stage2 = [remainder[i % len(remainder)] for i in range(n2)]
    assert_teacher_disjoint(stage2, reserved)
    return stage2, reserved


def build(a, fill_teacher_per_bench=0):
    """write stage-2 and stage-3 group files for this pipeline; returns their paths"""
    tr = [b for b in BENCH if b != a.fold]; d = wd(a); os.makedirs(d, exist_ok=True)
    per = PER_STAGE // len(tr); n_gold = int(round(PER_STAGE * a.r)) // len(tr); syn_src = {"yi": "tr_w20.jsonl", "wu": "tr_w20.jsonl", "bing": "trx.jsonl"}.get(a.arm)
    s2, s3 = [], []; exposure = {"teacher": {}, "gold": {}}
    for b in tr:
        t = source("teacher", b, a.seed); n2 = per + fill_teacher_per_bench
        # jiap follows jia's data recipe (spec section 5), but never receives compute fill.
        t2, t3 = teacher_blocks(t, n2, per - n_gold if a.arm in ("jia", "jiap") else 0, a.seed)
        s2 += t2; s3 += t3; exposure["teacher"][b] = exposure_counts(t2, t3)
        gold = take(source("gold", b, a.seed), n_gold, 0, a.seed)
        s3 += gold; exposure["gold"][b] = exposure_counts([], gold)
    if syn_src:
        s2 += take(syn_src, PER_STAGE, 0, a.seed); s3 += take(syn_src, PER_STAGE - n_gold * len(tr), PER_STAGE, a.seed)
    assert_teacher_disjoint(s2, s3)
    random.Random(a.seed).shuffle(s2); random.Random(a.seed + 1).shuffle(s3)
    open(f"{d}/s2.jsonl", "w").writelines(s2); open(f"{d}/s3.jsonl", "w").writelines(s3)
    json.dump({"identity": "sha256(qid, query, sorted docs)", "teacher_overlap": 0, "group_exposures": exposure},
              open(f"{d}/exposures.json", "w"), indent=1)
    return f"{d}/s2.jsonl", f"{d}/s3.jsonl"


def s1_flops(arm, seed):
    if arm == "yi": return flops("tr_w20.jsonl", 4, 4, seed, 2) + flops("tr_d3_e.jsonl", 2, 4, seed, 1, 0.5)
    if arm == "bing": return flops("trx.jsonl", 4, 4, seed, 2)
    return 0.0


def yi_C(a):
    """compute budget C(fold, seed, r): yi's planned full path (stage 1 + stage 2 + stage 3), from yi's own deterministic files"""
    y = argparse.Namespace(**{**vars(a), "arm": "yi"}); s2, s3 = build(y)
    return s1_flops("yi", a.seed) + flops(s2, 4, 4, a.seed) + flops(s3, 4, 4, a.seed, 1, 0.5)


def plan(a):
    d = wd(a); C = yi_C(a); s2, s3 = build(a); fill = 0
    me = lambda: s1_flops(a.arm, a.seed) + flops(s2, 4, 4, a.seed) + flops(s3, 4, 4, a.seed, 1, 0.5)
    cur = me()
    if a.arm in ("jia", "wu", "bing"):
        for _ in range(12):   # add teacher groups (equal per training benchmark) to stage 2 until within +-5% of C
            if abs(cur - C) <= 0.05 * C: break
            per_group = flops(s2, 4, 4, a.seed) / max(1, len(lines(s2)))
            fill = max(0, fill + int((C - cur) / per_group / 4)); s2, s3 = build(a, fill); cur = me()
        if abs(cur - C) > 0.05 * C: sys.exit(f"FAIL compute match: arm {a.arm} {cur:.3e} vs C {C:.3e}")
    p = {"arm": a.arm, "fold": a.fold, "seed": a.seed, "r": a.r, "C_yi": C, "planned_flops": cur, "ratio": cur / C, "fill_teacher_per_bench": fill,
         "s2": s2, "s3": s3, "s2_md5": md5(s2), "s3_md5": md5(s3), "s2_rows": len(lines(s2)), "s3_rows": len(lines(s3)),
         **json.load(open(f"{d}/exposures.json"))}
    json.dump(p, open(f"{d}/plan.json", "w"), indent=1); print(json.dumps(p)); return p


def torchrun(gpus, out, args, log, master_port=None):
    n = len(gpus.split(","))
    devices = gpus.replace(',', '' if master_port is None else '-')
    port = 29500 + random.randint(0, 400) if master_port is None else master_port
    sh(f"docker run --rm --name rtm-s23-{os.path.basename(out)}-{devices} --gpus '\"device={gpus}\"' --ipc=host --network host "
       f"-v {H}:/w -v {MODELS}:/models:ro --entrypoint torchrun {IMG} --nproc_per_node {n} --master_port {port} "
       f"/w/train.py {' '.join(args)} --out /w/{out}", log=log)


def stage23_gpus(gpus):
    devices = gpus.split(",")
    if (len(devices) != 4 or any(not g.isascii() or not g.isdigit() for g in devices)
            or len({int(g) for g in devices}) != 4):
        raise ValueError("stage 2/3 require four distinct GPU IDs with accum 4 (16 groups/step)")
    return devices


def candidate_steps(rows):
    total = (rows // 4) // 4  # train.py floor rule, world=4, epochs=1, accum=4
    if total < 2: raise ValueError("at least two optimizer steps required for midpoint/end selection")
    return [total // 2, total]


def consumed_rows(path, seed, steps):
    """Exact train.py sampler, for actual per-group exposures (including floor truncation)."""
    rows = lines(path); random.Random(seed).shuffle(rows); consumed = []
    for rank in range(4):
        mine = rows[rank::4]; rng = random.Random(seed + rank); order = []
        if not mine: raise ValueError("empty training shard")
        while len(order) < steps * 4: order += rng.sample(range(len(mine)), len(mine))
        consumed.extend(mine[j] for j in order[:steps * 4])
    return consumed


def checkpoint_candidates(ckdir, steps):
    """Validate BOTH candidates before serving either; a missing midpoint is a hard failure."""
    if len(steps) != 2 or not 0 < steps[0] < steps[1]: raise ValueError("expected distinct midpoint/end steps")
    logs = [json.loads(l) for l in lines(f"{ckdir}/compute_steps.jsonl")]
    if [r["step"] for r in logs] != list(range(1, steps[1] + 1)):
        raise ValueError("compute_steps.jsonl does not record the complete planned stage")
    result = []
    for index, step in enumerate(steps):
        ck = f"{ckdir}/s{step}"; root_end = False
        if not os.path.isdir(ck):
            if index == 0: raise ValueError(f"missing midpoint checkpoint: {ck}")
            ck = ckdir; root_end = True
        args = json.load(open(f"{ck}/train_args.json"))
        if (args.get("checkpoint_step") != step or args.get("steps_total") != steps[1]
                or args.get("world") != 4 or args.get("accum") != 4 or args.get("epochs") != 1):
            raise ValueError(f"checkpoint recorded step/configuration mismatch: {ck}, expected step {step}")
        if not os.path.isfile(f"{ck}/config.json"):
            raise ValueError(f"missing model config: {ck}")
        if os.path.isfile(f"{ck}/model.safetensors.index.json"):
            shards = set(json.load(open(f"{ck}/model.safetensors.index.json"))["weight_map"].values())
            complete = bool(shards) and all(os.path.isfile(f"{ck}/{s}") for s in shards)
        else:
            complete = os.path.isfile(f"{ck}/model.safetensors")
        if not complete: raise ValueError(f"missing checkpoint weights: {ck}")
        result.append({"path": ck, "step": step, "recorded_step": args["checkpoint_step"], "root_as_end": root_end})
    return result


def serve(g, port, ckpt):
    sh(f"bash serve_sel.sh rtm-s23ev-{g} {g} {port} {os.path.join(H, ckpt)}")
    for _ in range(90):
        if subprocess.run(f"curl -sf -m3 localhost:{port}/v1/models | grep -q '\"id\":\"sel\"'", shell=True).returncode == 0: return
        time.sleep(10)
    sys.exit(f"FAIL serve {ckpt}")


def rank(g, port, ckpt, qf, pool, tag, log):
    serve(g, port, ckpt)
    try:
        if os.path.exists(f"ctx_{tag}.jsonl"): os.remove(f"ctx_{tag}.jsonl")
        sh(f"K1_BUDGETS={B5} python3 infer.py {qf} --pooldir {pool} --port {port} --tag {tag} --cond 0 --par 16", log=log)
    finally: subprocess.run(f"docker rm -f rtm-s23ev-{g}", shell=True, stdout=subprocess.DEVNULL)


def select(a, stage, ckdir, steps, gpus, log):
    """inner-area selection among the checkpoint candidates (midpoint, final); ties -> earlier"""
    tr = [b for b in BENCH if b != a.fold]; g = stage23_gpus(gpus); res = []
    candidates = checkpoint_candidates(ckdir, steps)
    for candidate in candidates:
        ck, st = candidate["path"], candidate["step"]
        tag = f"S23_{a.arm}_{a.fold}_s{a.seed}_r{a.r}_st{stage}_{st}"
        if getattr(a, "smoke_steps", 0): tag += "_smoke50"
        for b in tr: rank(g[0], 18480 + int(g[0]), ck, f"q_innersel_{b}.json", "pool_s23", f"{tag}_inner_{b}", log)
        out = subprocess.run(["python3", "inner_eval.py", tag, ",".join(tr), "--sel"], capture_output=True, text=True)
        if out.returncode: sys.exit("FAIL inner_eval: " + out.stdout + out.stderr)
        res.append({**candidate, "area": json.loads(out.stdout.strip().splitlines()[-1])["area"]})
    best = max(res, key=lambda r: (r["area"], -r["step"]))
    return best["path"], res


def run(a):
    stage23_gpus(a.gpus)  # refuse before file reads, docker, or any training/serving launch
    p = json.load(open(f"{wd(a)}/plan.json")); d = wd(a)
    smoke = getattr(a, "smoke_steps", 0)
    if smoke not in (0, 50): raise ValueError("smoke test is 50 steps per stage")
    if smoke: d += "/smoke50"; os.makedirs(d, exist_ok=True)
    log = f"{d}/run.log"
    if md5(p["s2"]) != p["s2_md5"] or md5(p["s3"]) != p["s3_md5"]: sys.exit("FAIL group files changed since plan")
    assert_teacher_disjoint(lines(p["s2"]), lines(p["s3"]))
    if p.get("teacher_overlap") != 0 or "group_exposures" not in p:
        raise ValueError("plan lacks v3.4 identity/exposure records; regenerate the plan")
    start = {"yi": f"/w/ckpt/s23_s1_yi_s{a.seed}" if a.seed else "/w/ckpt/d3_e", "bing": f"/w/ckpt/s23_s1_bing_s{a.seed}"}.get(a.arm, A0)
    steps2, steps3 = candidate_steps(p["s2_rows"]), candidate_steps(p["s3_rows"])
    extra = []
    if smoke:
        if min(steps2[-1], steps3[-1]) < smoke: raise ValueError("planned stage shorter than smoke test")
        steps2 = steps3 = [smoke // 2, smoke]; extra = ["--max-steps", str(smoke)]
    torchrun(a.gpus, f"{d}/st2", ["--model", start, "--data", f"/w/{p['s2']}", "--seed", str(a.seed), "--save-at", ",".join(map(str, steps2))] + TRAIN_COMMON + extra, log)
    ck2, c2 = select(a, 2, f"{d}/st2", steps2, a.gpus, log)
    torchrun(a.gpus, f"{d}/st3", ["--model", "/w/" + ck2, "--data", f"/w/{p['s3']}", "--seed", str(a.seed), "--w-base", "0.5", "--save-at", ",".join(map(str, steps3))] + TRAIN_COMMON + extra, log)
    ck3, c3 = select(a, 3, f"{d}/st3", steps3, a.gpus, log)
    qf, pool, nm = DEVSET[a.fold]; g = a.gpus.split(",")[0]; tag = f"S23_{a.arm}_{a.fold}_s{a.seed}_r{a.r}_heldout_{nm}"
    if smoke: tag += "_smoke50"
    rank(g, 18480 + int(g), ck3, qf, pool, tag, log)
    sampled2, sampled3 = consumed_rows(p["s2"], a.seed, steps2[-1]), consumed_rows(p["s3"], a.seed, steps3[-1])
    # group_exposures in the plan count source-row allocations; these count the actual
    # full training paths, including dropped tails and the optional smoke step cap.
    actual = {kind: exposure_counts([r for r in sampled2 if bool(json.loads(r).get("nolabel")) == teacher],
                                   [r for r in sampled3 if bool(json.loads(r).get("nolabel")) == teacher])
              for kind, teacher in (("teacher", True), ("labelled", False))}
    json.dump({**p, "smoke_steps": smoke, "training_group_exposures": actual,
               "stage2_candidates": c2, "stage2_selected": ck2, "stage3_candidates": c3, "stage3_selected": ck3, "heldout_ctx": f"ctx_{tag}.jsonl"},
              open(f"{d}/results.json", "w"), indent=1)
    print("S23DONE", d)


def s1(a):
    # W20 must use four distinct devices; the continuation uses exactly the first two.
    g = stage23_gpus(a.gpus)
    if a.arm not in ("yi", "bing"): raise ValueError("stage 1 supports yi/bing")
    if getattr(a, "smoke", False) or getattr(a, "smoke_steps", 0): raise ValueError("stage 1 cannot produce a smoke starting checkpoint")
    preflight(a, stage1=True)
    out = f"ckpt/s23_s1_{a.arm}_s{a.seed}"; log = f"s23/s1_{a.arm}_s{a.seed}.log"; os.makedirs("s23", exist_ok=True)
    base = ["--lr", "1e-5", "--max-q", "4000", "--max-d", "1100", "--w-bce", "0.2",
            "--w-kl", "0", "--group", "16", "--tau", "1", "--gamma", "0", "--shuffle-docs", "1",
            "--accum", "4", "--seed", str(a.seed)]
    if a.arm == "yi":
        torchrun(",".join(g[:4]), out + "_w20", ["--model", A0, "--data", "/w/tr_w20.jsonl", "--epochs", "2"] + base, log)
        from preflight_s23 import checkpoint_recipe, training_rows
        checkpoint_recipe(os.path.join(H, out + "_w20"), a.seed, 4, 2, "/w/tr_w20.jsonl", A0,
                          training_rows("tr_w20.jsonl", 17784))
        torchrun(",".join(g[:2]), out, ["--model", f"/w/{out}_w20", "--data", "/w/tr_d3_e.jsonl", "--epochs", "1", "--w-base", "0.5"] + base, log)
        checkpoint_recipe(os.path.join(H, out), a.seed, 2, 1, "/w/tr_d3_e.jsonl", f"/w/{out}_w20",
                          training_rows("tr_d3_e.jsonl", 1500))
    elif a.arm == "bing":
        torchrun(",".join(g[:4]), out, ["--model", A0, "--data", "/w/trx.jsonl", "--epochs", "2"] + base, log)
    print("S1DONE", out)



# v4.1b all-data grid. Legacy functions above retain the audited fold path.
S2_LEVELS = (8000, 16000)
S3_CELLS = ((0.25, 4000), (0.25, 8000), (0.5, 4000), (0.5, 8000))


def clone(a, **kw): return argparse.Namespace(**{**vars(a), **kw})


def gpu_sets(a):
    first = stage23_gpus(a.gpus)
    second = getattr(a, "gpus2", None)
    if second is None: return [a.gpus]
    other = stage23_gpus(second)
    if set(map(int, first)) & set(map(int, other)):
        raise ValueError("--gpus2 must be disjoint from --gpus")
    return [a.gpus, second]


def gpu_map(a, fn, jobs, scoring=False):
    """Barrier between batches: one job per GPU/set, results in input order.

    Training writes separate output directories. Scoring writes per-tag files;
    inner_rank/eval have no mutable shared state, cwd changes or shared sessions
    (k1._post uses request-local urllib objects). Tags must be unique per job.
    """
    sets = gpu_sets(a)
    if len(sets) == 1: return [fn(a, *job) for job in jobs]
    workers = ([clone(a, _score_gpu=str(int(g))) for s in sets for g in s.split(",")]
               if scoring else [clone(a, gpus=s) for s in sets])
    results = []
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        for start in range(0, len(jobs), len(workers)):
            futures = [pool.submit(fn, worker, *job) for worker, job in zip(workers, jobs[start:start + len(workers)])]
            results.extend(f.result() for f in futures)
    return results


def validate_train_files():
    """Exact frozen train identities, not substring matching or a dev evaluation list."""
    fr = json.load(open("DATA_FREEZE_STAGE23.json"))["bench"]
    for b in BENCH:
        ids = _question_ids(f"q_train_{b}.json")
        train, inner = set(fr[b]["train"]), set(fr[b]["inner"])
        if train & inner or len(ids) != len(set(ids)) or set(ids) != train or set(ids) & inner:
            raise ValueError(f"{b}: q_train identities differ from frozen train-minus-inner")
    return fr


def all_benchmark_rows(a, b, fill_teacher_per_bench=0):
    """The grid's exact per-benchmark recipe, including source-byte seeded cycling."""
    real2 = a.s2_groups // 2 // 5
    gold3 = int(a.s3_groups * a.r) // 5
    teacher3 = a.s3_groups // 5 - gold3 if a.arm == "jia" else 0
    reserve = max(int(n * (1 - r)) // 5 for r, n in S3_CELLS) if a.arm == "jia" else 0
    teacher = source("teacher", b, a.seed, a.teacher, a.teacher_config, True)
    t2, reserved = teacher_blocks(teacher, real2 + fill_teacher_per_bench, reserve, a.seed)
    gold = source("gold", b, a.seed, a.teacher, a.teacher_config, True)
    return t2, reserved[:teacher3], take(gold, gold3, seed=a.seed), (teacher, gold)


def build_all(a, fill_teacher_per_bench=0):
    if getattr(a, "cmd", None) in ("budget", "lobo"):
        return build_budget(a, fill_teacher_per_bench)
    if a.arm not in ("yi", "jia"): raise ValueError("all-data supports yi and jia")
    if a.s2_groups not in S2_LEVELS or (a.r, a.s3_groups) not in S3_CELLS: raise ValueError("invalid grid cell")
    if fill_teacher_per_bench < 0: raise ValueError("negative teacher fill")
    validate_train_files()
    d = wd(a); os.makedirs(d, exist_ok=True)
    gold3 = int(a.s3_groups * a.r) // 5
    # One stage-2 run per level: reserve the union (largest prefix) needed by its four children.
    reserve = max(int(n * (1 - r)) // 5 for r, n in S3_CELLS) if a.arm == "jia" else 0
    s2, s3 = [], []; exposure = {"teacher": {}, "gold": {}}
    for b in BENCH:
        t2, t3, gold, _ = all_benchmark_rows(a, b, fill_teacher_per_bench)
        s2 += t2; s3 += t3; exposure["teacher"][b] = exposure_counts(t2, t3)
        s3 += gold; exposure["gold"][b] = exposure_counts([], gold)
    if a.arm == "yi":
        # Disjoint, without replacement from the same seeded Crucible permutation.
        syn = take("tr_w20.jsonl", len(lines("tr_w20.jsonl")), seed=a.seed)
        n2 = a.s2_groups // 2; n3 = a.s3_groups - gold3 * 5
        if n2 + n3 > len(syn): raise ValueError("insufficient nonoverlapping Crucible groups")
        s2 += syn[:n2]; s3 += syn[n2:n2 + n3]
    assert_teacher_disjoint(s2, s3)
    random.Random(a.seed).shuffle(s2); random.Random(a.seed + 1).shuffle(s3)
    p2, p3 = f"{d}/s2.jsonl", f"{d}/s3.jsonl"
    with open(p2, "w") as f: f.writelines(s2)
    with open(p3, "w") as f: f.writelines(s3)
    json.dump({"identity": "sha256(qid, query, sorted docs)", "teacher_overlap": 0,
               "teacher_reserved_per_bench": reserve, "group_exposures": exposure}, open(f"{d}/exposures.json", "w"), indent=1)
    return p2, p3


def reuse_plans(cells):
    """Reuse frozen plan.json files from `s23.py plan` instead of re-running the FLOPs search.

    Accepted only when every cell has a plan with matching identity fields, one shared fill,
    and group files whose md5 equals the plan (rebuilt deterministically at the frozen fill
    if a file was overwritten). Otherwise returns None and the caller plans from scratch.
    """
    if any(getattr(c, "smoke", False) for c in cells): return None
    plans = []
    for c in cells:
        f = f"{wd(c)}/plan.json"
        if not os.path.isfile(f): return None
        p = json.load(open(f))
        want = {"arm": c.arm, "all_data": True, "seed": c.seed, "s2_groups": c.s2_groups,
                "s3_groups": c.s3_groups, "r": c.r, "teacher": c.teacher,
                "teacher_config": c.teacher_config, "data_freeze_md5": md5("DATA_FREEZE_STAGE23.json")}
        if any(p.get(k) != v for k, v in want.items()): return None
        fill, ratio = p.get("fill_teacher_per_bench"), p.get("ratio")
        if p.get("version") != "4.1b" or type(fill) is not int or fill < 0: return None
        if c.arm == "yi" and fill != 0: return None
        if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or abs(ratio - 1) > .05: return None
        if p.get("s2") != f"{wd(c)}/s2.jsonl" or p.get("s3") != f"{wd(c)}/s3.jsonl": return None
        plans.append(p)
    if len({p["fill_teacher_per_bench"] for p in plans}) != 1: return None
    if len({p["s2_md5"] for p in plans}) != 1: return None
    for c, p in zip(cells, plans):
        # Always rebuild through build_all (source manifests and teacher scores re-validated),
        # then require byte-identical groups; any drift fails closed.
        build_all(c, p["fill_teacher_per_bench"])
        if (md5(p["s2"]) != p["s2_md5"] or md5(p["s3"]) != p["s3_md5"]
                or len(lines(p["s2"])) != p["s2_rows"] or len(lines(p["s3"])) != p["s3_rows"]):
            raise ValueError(f"rebuilt group files differ from frozen plan: {wd(c)}")
        print("REUSE_PLAN", wd(c), "fill", p["fill_teacher_per_bench"], "ratio", round(p["ratio"], 4), flush=True)
    return plans


def plan_level(a):
    """Freeze four full-path budgets with one common stage-2 file and fill for the level.
    Search the intersection of all four ±5% intervals. Costs use exact planned samplers.
    """
    cells = ([a] if getattr(a, "smoke", False) else [clone(a, r=r, s3_groups=n) for r, n in S3_CELLS])
    reused = reuse_plans(cells)
    if reused: return reused
    budgets = []
    for c in cells:
        y = clone(c, arm="yi"); p2, p3 = build_all(y)
        budgets.append(s1_flops("yi", a.seed) + flops(p2, 4, 4, a.seed) + flops(p3, 4, 4, a.seed, 1, 0.5))
    def measure(fill):
        result = []
        for c in cells:
            p2, p3 = build_all(c, fill)
            result.append((p2, p3, s1_flops(a.arm, a.seed) + flops(p2, 4, 4, a.seed) + flops(p3, 4, 4, a.seed, 1, 0.5)))
        return result
    fill = 0; result = measure(fill)
    if a.arm == "jia":
        # Grow to the lower bound, then bisect group count; do not cycle the reserved block.
        def low(rows): return all(v >= .95 * C for (_, _, v), C in zip(rows, budgets))
        lo, hi = -1, 0
        while not low(result):
            lo, hi = hi, max(1, hi * 2)
            if hi > 10000000: raise ValueError("unable to bracket teacher FLOPs fill")
            result = measure(hi)
        while hi - lo > 1:
            mid = (lo + hi) // 2; trial = measure(mid)
            if low(trial): hi = mid
            else: lo = mid
        fill = hi; result = measure(fill)
    if any(abs(v / C - 1) > .05 for (_, _, v), C in zip(result, budgets)):
        raise ValueError("no shared stage-2 fill matches all four full-path cells within ±5%; plans not frozen")
    plans = []
    if len({md5(p2) for p2, _, _ in result}) != 1: raise ValueError("stage-2 grid children differ")
    for c, C, (p2, p3, value) in zip(cells, budgets, result):
        p = {"version": "4.1b", "arm": c.arm, "all_data": True, "seed": c.seed,
             "s2_groups": c.s2_groups, "s3_groups": c.s3_groups, "r": c.r,
             "teacher": c.teacher, "teacher_config": c.teacher_config,
             "budget_key": [c.seed, c.s2_groups, c.r, c.s3_groups], "C_yi": C,
             "planned_flops": value, "ratio": value / C, "fill_teacher_per_bench": fill,
             "s2": p2, "s3": p3, "s2_md5": md5(p2), "s3_md5": md5(p3),
             "s2_rows": len(lines(p2)), "s3_rows": len(lines(p3)),
             "data_freeze_md5": md5("DATA_FREEZE_STAGE23.json"),
             **json.load(open(f"{wd(c)}/exposures.json"))}
        json.dump(p, open(f"{wd(c)}/plan.json", "w"), indent=1); plans.append(p)
    return plans


def inner_score(a, ckpt, tag, sel=True, options=None):
    """All-data model decisions always use inner questions; never DEVSET."""
    from inner_rank import rank_inner
    from inner_eval import evaluate
    options = options or {"neighbour_context": False, "rrf_weight": 0.0}
    token = a.teacher.replace("+", "_") + ("_" + md5(a.teacher_config)[:12] if a.teacher_config else "")
    tag = f"ALL_{token}_{tag}"
    if a.smoke_steps and "_smoke50" not in tag: tag += "_smoke50"
    g = getattr(a, "_score_gpu", None) or stage23_gpus(a.gpus)[0]; port = 18480 + int(g)
    serve(g, port, ckpt)
    try:
        smoke = getattr(a, "smoke", False)
        rank_inner(tag, BENCH, sel, str(port), smoke=smoke, **options)
        return evaluate(tag, BENCH, sel, smoke=smoke)
    finally:
        subprocess.run(["docker", "rm", "-f", f"rtm-s23ev-{g}"], stdout=subprocess.DEVNULL)


def all_start(a):
    return (f"/w/ckpt/s23_s1_yi_s{a.seed}" if a.seed else "/w/ckpt/d3_e") if a.arm == "yi" else A0


def train_all_stage(a, p, stage, start, out):
    if md5(p[f"s{stage}"]) != p[f"s{stage}_md5"]: raise ValueError("group file changed since plan")
    if md5("DATA_FREEZE_STAGE23.json") != p["data_freeze_md5"]: raise ValueError("data freeze changed since plan")
    assert_teacher_disjoint(lines(p["s2"]), lines(p["s3"]))
    steps = candidate_steps(p[f"s{stage}_rows"])
    extra = []
    if a.smoke_steps:
        if steps[-1] < 50: raise ValueError("stage shorter than smoke test")
        steps = [25, 50]; extra = ["--max-steps", "50"]
    done = out + "/training.json"
    if getattr(a, "cmd", None) in ("budget", "lobo") and not a.smoke_steps and os.path.isfile(done):
        # Resume after an interrupted budget run: training.json is written only after a completed stage.
        prev = json.load(open(done))
        if prev.get("stage") == stage and prev.get("planned_full_path") == p:
            ok = {c["path"] for c in checkpoint_candidates(out, steps)}
            if ok == {c["path"] for c in prev["candidates"]}:
                print("RESUME_STAGE", stage, out, flush=True)
                return prev["candidates"]
    os.makedirs(out, exist_ok=True)
    args = ["--model", start, "--data", "/w/" + p[f"s{stage}"], "--seed", str(a.seed),
            "--save-at", ",".join(map(str, steps))] + TRAIN_COMMON + extra
    if stage == 3: args += ["--w-base", "0.5"]
    if getattr(a, "gpus2", None) or getattr(a, "cmd", None) in ("budget", "lobo"):
        # Disjoint sets have distinct minima. Delimited device IDs also keep
        # concurrent container names unambiguous; train.py arguments are unchanged.
        torchrun(a.gpus, out, args, out + "/run.log", master_port=29500 + min(map(int, a.gpus.split(","))))
    else:
        torchrun(a.gpus, out, args, out + "/run.log")
    candidates = checkpoint_candidates(out, steps)
    # Actual selected-prefix cost is reported separately from planned complete-path C.
    logs = [json.loads(l) for l in lines(out + "/compute_steps.jsonl")]
    for c in candidates:
        c["actual_flops"] = logs[c["step"] - 1]["est_flops"]
    sampled = consumed_rows(p[f"s{stage}"], a.seed, steps[-1])
    json.dump({"smoke_steps": a.smoke_steps, "stage": stage, "planned_full_path": p,
               "actual_group_exposures": exposure_counts(sampled if stage == 2 else [], sampled if stage == 3 else []),
               "candidates": candidates}, open(out + "/training.json", "w"), indent=1)
    return candidates


def grid_root(a):
    # Separate teacher/config namespaces and smoke choices; no production freeze from smoke.
    root = all_namespace(a)
    return root + (f"/smoke50/cell_g2{a.s2_groups}_r{a.r}_g3{a.s3_groups}"
                   if a.smoke_steps or getattr(a, "smoke", False) else "")


def choice_file(a):
    suffix = f"_s{a.seed}" if getattr(a, "smoke", False) else ""
    return f"{grid_root(a)}/choice_{a.arm}{suffix}.json"


def score_candidates(a, candidates, prefix, sel=True):
    result = []
    jobs = [(c["path"], f"{prefix}_{index}", sel) for index, c in enumerate(candidates)]
    for c, score in zip(candidates, gpu_map(a, inner_score, jobs, scoring=True)):
        result.append({**c, "area": score["area"], "evaluation": score})
    return sorted(result, key=lambda c: (-c["area"], c["step"], c["s2_groups"], c.get("s3_groups", 0), c.get("r", 0)))


def select_options(a, ckpt):
    """Greedy full-inner options, each retained only for >=1 area point; yi only."""
    current = {"neighbour_context": False, "rrf_weight": 0.0}; history = []
    def scores(options, labels):
        values = gpu_map(a, inner_score, [(ckpt, f"options_yi_{label}", False, opt)
                                         for opt, label in zip(options, labels)], scoring=True)
        history.extend({**opt, "area": v["area"]} for opt, v in zip(options, values))
        return [v["area"] for v in values]
    trial = {**current, "neighbour_context": True}
    area, value = scores([current, trial], ["base", "neighbours"])
    if value >= area + 1: current, area = trial, value
    weights = (.25, .5, .75, 1.0)
    opts = [{**current, "rrf_weight": w} for w in weights]
    trials = list(zip(scores(opts, [f"rrf{w}" for w in weights]), weights, opts))
    value, _, opt = max(trials, key=lambda t: (t[0], -t[1]))
    if value >= area + 1: current = opt
    return {"options": current, "trials": history, "min_gain_points": 1}


def load_choice(a):
    choice = json.load(open(choice_file(a)))
    if (choice.get("smoke") or choice.get("version") != "4.1b" or choice.get("protocol") != PROTOCOL or choice["arm"] != a.arm or choice["seed"] != 0
            or choice["teacher"] != a.teacher or choice["teacher_config_md5"] != (md5(a.teacher_config) if a.teacher_config else None)
            or choice["data_freeze_md5"] != md5("DATA_FREEZE_STAGE23.json") or choice["smoke_steps"] != a.smoke_steps):
        raise ValueError("incompatible frozen grid choice")
    return choice


def grid(a):
    gpu_sets(a)
    validate_train_files()
    if not a.all_data or a.arm not in ("yi", "jia"): raise ValueError("grid requires --all-data and yi/jia")
    if a.seed not in (0, 1, 2): raise ValueError("expected seed 0, 1 or 2")
    if a.smoke_steps not in (0, 50): raise ValueError("smoke test is 50 steps per stage")
    if getattr(a, "smoke", False) or a.smoke_steps:
        a = clone(a, smoke=True, smoke_steps=50)
    counts = preflight(a)
    if getattr(a, "smoke", False):
        print("ESTIMATED_CANDIDATE_PAIRS", 4 * counts["smoke"],
              "(one cell, two checkpoints/stage, 10 INNERSEL questions/benchmark, top 600)", flush=True)
        return smoke_grid(a)
    if a.seed == 0 and os.path.exists(choice_file(a)): raise ValueError("choice already frozen; use a separate output directory for a new search")
    # Jia inherits the yi inference switches, but selects its own independent training grid.
    yi_choice = load_choice(clone(a, arm="yi", seed=0)) if a.arm == "jia" else None
    frozen = load_choice(a) if a.seed else None
    pairs = 0 if frozen else 12 * counts["selection"] + (9 if a.arm == "yi" else 3) * counts["full"]
    print("ESTIMATED_CANDIDATE_PAIRS", pairs, "(inner model selection/reporting; excludes teacher preparation)", flush=True)
    root = grid_root(a); os.makedirs(root, exist_ok=True)
    levels = (frozen["s2_groups"],) if frozen else S2_LEVELS
    plans = {level: plan_level(clone(a, s2_groups=level)) for level in levels}
    s2_candidates = []
    jobs2 = [(plans[level][0], 2, all_start(a), f"{root}/{a.arm}_s{a.seed}_g2{level}/st2") for level in levels]
    for level, candidates in zip(levels, gpu_map(a, train_all_stage, jobs2)):
        s2_candidates += [{**c, "s2_groups": level, "checkpoint": "mid" if i == 0 else "end"} for i, c in enumerate(candidates)]
    if frozen:
        c2 = next(c for c in s2_candidates if c["checkpoint"] == frozen["stage2_checkpoint"])
        ranked2 = s2_candidates
    else:
        ranked2 = score_candidates(a, s2_candidates, f"grid_{a.arm}_st2" + ("_smoke50" if a.smoke_steps else ""))
        c2 = ranked2[0]
    candidates3 = []
    cells = [p for p in plans[c2["s2_groups"]]
             if not frozen or (p["r"], p["s3_groups"]) == (frozen["r"], frozen["s3_groups"])]
    jobs3 = [(p, 3, "/w/" + c2["path"],
              f"{root}/{a.arm}_s{a.seed}_g2{c2['s2_groups']}_r{p['r']}_g3{p['s3_groups']}/st3") for p in cells]
    for p, cs in zip(cells, gpu_map(a, train_all_stage, jobs3)):
        candidates3 += [{**c, "s2_groups": c2["s2_groups"], "r": p["r"], "s3_groups": p["s3_groups"],
                         "checkpoint": "mid" if i == 0 else "end", "plan": p} for i, c in enumerate(cs)]
    if frozen:
        chosen = next(c for c in candidates3 if c["checkpoint"] == frozen["stage3_checkpoint"])
        ranked3, finalists = candidates3, []
        options = frozen["inference"]
    else:
        suffix = "_smoke50" if a.smoke_steps else ""
        ranked3 = score_candidates(a, candidates3, f"grid_{a.arm}_st3{suffix}")
        finalists = score_candidates(a, ranked3[:2], f"grid_{a.arm}_full{suffix}", False)
        chosen = finalists[0]
        options = yi_choice["inference"] if yi_choice else select_options(a, chosen["path"])
    # Report raw and enhanced inner results for both arms. These do not reselect a recipe.
    reported = {}
    if not frozen:
        reported["plain"] = chosen["evaluation"]
        reported["with_options"] = inner_score(a, chosen["path"], f"report_{a.arm}_s0" + ("_smoke50" if a.smoke_steps else ""),
                                               False, options["options"])
    result = {"version": "4.1b", "protocol": PROTOCOL, "arm": a.arm, "seed": a.seed, "teacher": a.teacher,
              "teacher_config_md5": md5(a.teacher_config) if a.teacher_config else None,
              "data_freeze_md5": md5("DATA_FREEZE_STAGE23.json"), "smoke_steps": a.smoke_steps,
              "s2_groups": c2["s2_groups"], "r": chosen["r"], "s3_groups": chosen["s3_groups"],
              "stage2_checkpoint": c2["checkpoint"], "stage3_checkpoint": chosen["checkpoint"],
              "stage2_candidates": ranked2, "stage3_candidates": ranked3, "full_inner_finalists": finalists,
              "stage2_selected": c2["path"], "stage3_selected": chosen["path"], "inference": options,
              "selected_actual_flops": s1_flops(a.arm, a.seed) + c2["actual_flops"] + chosen["actual_flops"],
              "selected_planned_flops": chosen["plan"]["planned_flops"], "inner_report": reported,
              "frozen_choice": choice_file(a), "reused_seed0_choice": bool(frozen)}
    atomic_json(f"{root}/results_{a.arm}_s{a.seed}.json", result)
    if not a.seed: atomic_json(choice_file(a), result)
    print("S23GRID_DONE", choice_file(a))
    return result


def smoke_grid(a):
    """One training run per stage; four small-inner evaluations, no full-inner or tuning."""
    if os.path.exists(choice_file(a)): raise ValueError("smoke choice already exists; use a separate prepared copy")
    root = grid_root(a); os.makedirs(root, exist_ok=True)
    p, = plan_level(a)
    cs2 = train_all_stage(a, p, 2, all_start(a), f"{root}/{a.arm}_s{a.seed}/st2")
    cs2 = [{**c, "s2_groups": a.s2_groups} for c in cs2]
    ranked2 = score_candidates(a, cs2, f"smoke_{a.arm}_s{a.seed}_st2")
    cs3 = train_all_stage(a, p, 3, "/w/" + ranked2[0]["path"], f"{root}/{a.arm}_s{a.seed}/st3")
    cs3 = [{**c, "s2_groups": a.s2_groups, "s3_groups": a.s3_groups, "r": a.r} for c in cs3]
    ranked3 = score_candidates(a, cs3, f"smoke_{a.arm}_s{a.seed}_st3")
    result = {"version": "4.1b", "protocol": PROTOCOL, "smoke": True, "smoke_steps": 50,
              "arm": a.arm, "seed": a.seed, "s2_groups": a.s2_groups, "s3_groups": a.s3_groups, "r": a.r,
              "stage2_candidates": ranked2, "stage3_candidates": ranked3, "full_inner_finalists": [],
              "stage2_selected": ranked2[0]["path"], "stage3_selected": ranked3[0]["path"],
              "inference": {"options": {"neighbour_context": False, "rrf_weight": 0.0}, "trials": []},
              "plan": p, "production_choice": False}
    atomic_json(f"{root}/results_{a.arm}_s{a.seed}.json", result)
    # Kept apart from production and never accepted by load_choice.
    atomic_json(choice_file(a), result)
    print("S23SMOKE_DONE", f"{root}/results_{a.arm}_s{a.seed}.json")
    return result


def atomic_json(path, value):
    folder = os.path.dirname(path); os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="choice_", suffix=".tmp", dir=folder); os.close(fd)
    try:
        with open(tmp, "w") as f: json.dump(value, f, indent=1)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.remove(tmp)


def budget_dir(a):
    if getattr(a, "cmd", None) == "lobo":
        root = f"{all_namespace(a)}/lobo/{a.arm}_{a.fold}_s{a.seed}"
        return root + ("/smoke50" if a.smoke_steps else "")
    return f"{all_namespace(a)}/budget/{a.arm}_f{a.frac}_s{a.seed}"


def budget_write(path, rows):
    """Immutable base supervision bytes, shared identically by yi and jia-small."""
    from pathlib import Path
    content = "".join(rows)
    if os.path.exists(path):
        if Path(path).read_text() != content:
            raise ValueError(f"frozen budget groups changed: {path}")
    else:
        folder = os.path.dirname(path); os.makedirs(folder, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="groups_", suffix=".tmp", dir=folder)
        try:
            with os.fdopen(fd, "w") as f: f.write(content)
            try: os.link(temporary, path)
            except FileExistsError:
                if Path(path).read_text() != content:
                    raise ValueError(f"frozen budget groups changed: {path}")
        finally:
            os.unlink(temporary)
    return path


def budget_choices(a):
    choice_args = clone(a, smoke_steps=0, smoke=False) if a.cmd == "lobo" else a
    choices = {arm: load_choice(clone(choice_args, arm=arm, seed=0)) for arm in ("yi", "jia")}
    for arm, choice in choices.items():
        if (choice["s2_groups"] != {"yi": 8000, "jia": 16000}[arm]
                or choice["r"] != .5 or choice["s3_groups"] != 8000
                or choice["stage3_checkpoint"] != "end"
                or choice["stage2_checkpoint"] not in ("mid", "end")):
            raise ValueError(f"{arm}: frozen choice differs from SPEC_BUDGET recipe")
    return choices


def prepare_budget(a, sample):
    """Budget caps its allocations; LOBO reuses the full yi grid recipe exactly."""
    from budget_sample import distinct_rows, freeze_json, supervision_counts
    validate_train_files()
    d = budget_dir(a); os.makedirs(d, exist_ok=True)
    # LOBO's three arms share the same unique real supervision. Its matched
    # baseline may only repeat that block; budget retains its older jia recipe.
    desired2 = 8000 if a.arm == "jia" and a.cmd != "lobo" else 4000
    per2 = math.floor(desired2 * a.frac / 5)
    per3 = math.floor(4000 * a.frac / 5)
    data = {"teacher2": {}, "teacher3": {}, "gold3": {}, "syn2": [], "syn3": []}
    choice_args = clone(a, smoke_steps=0, smoke=False) if a.cmd == "lobo" else a
    inputs = {"DATA_FREEZE_STAGE23.json": md5("DATA_FREEZE_STAGE23.json"),
              **{choice_file(clone(choice_args, arm=arm, seed=0)): md5(choice_file(clone(choice_args, arm=arm, seed=0)))
                 for arm in ("yi", "jia")}}
    allowed = set().union(*(set(v["qids"]) for v in sample["bench"].values()))
    # Real and synthetic qid namespaces must be distinguishable in supervision counts.
    all_real = set().union(*(_question_ids(f"q_train_{b}.json", unique=True) for b in BENCH))
    if a.cmd == "lobo":
        frozen = json.load(open("DATA_FREEZE_STAGE23.json"))["bench"]
        all_real.update(q for b in BENCH for q in frozen[b]["inner"])
    for b in BENCH:
        keep = set(sample["bench"][b]["qids"])
        if not keep and a.cmd == "lobo" and b == a.fold:
            for kind in ("teacher2", "teacher3", "gold3"):
                data[kind][b] = []
            continue
        if a.cmd == "lobo":
            recipe = clone(a, arm="yi", s2_groups=8000, r=.5, s3_groups=8000)
            t2, t3, g3, sources = all_benchmark_rows(recipe, b)
            if (len(t2), len(t3), len(g3)) != (800, 0, 800):
                raise ValueError(f"{b}: LOBO counts differ from full grid recipe")
            if any(json.loads(r)["qid"] not in keep for r in t2 + t3 + g3):
                raise ValueError(f"{b}: LOBO source escaped frozen question set")
            data["teacher2"][b], data["teacher3"][b], data["gold3"][b] = t2, t3, g3
            inputs.update({path: md5(path) for path in sources})
            continue
        teacher = source("teacher", b, a.seed, a.teacher, a.teacher_config, True)
        gold = source("gold", b, a.seed, a.teacher, a.teacher_config, True)
        inputs[teacher], inputs[gold] = md5(teacher), md5(gold)
        eligible = distinct_rows([r for r in lines(teacher) if json.loads(r)["qid"] in keep], group_identity)
        if not eligible: raise ValueError(f"{b}: no budget teacher groups")
        path = budget_write(f"{d}/eligible_teacher_{b}.jsonl", eligible)
        # Reserve stage-3 teacher identities first, leaving at least one for stage 2.
        n3 = min(per3, max(0, len(eligible) - 1)) if a.arm == "jia" and a.cmd != "lobo" else 0
        n2 = min(per2, len(eligible) - n3)
        if not n2: raise ValueError(f"{b}: no stage-2 budget teacher allocation")
        t2, t3 = teacher_blocks(path, n2, n3, a.seed)
        data["teacher2"][b], data["teacher3"][b] = t2, t3
        eligible_gold = distinct_rows([r for r in lines(gold) if json.loads(r)["qid"] in keep], group_identity)
        if not eligible_gold: raise ValueError(f"{b}: no budget gold groups")
        path = budget_write(f"{d}/eligible_gold_{b}.jsonl", eligible_gold)
        data["gold3"][b] = take(path, min(per3, len(eligible_gold)), seed=a.seed)
    if a.arm == "yi":
        inputs["tr_w20.jsonl"], inputs["tr_d3_e.jsonl"] = md5("tr_w20.jsonl"), md5("tr_d3_e.jsonl")
        syn = take("tr_w20.jsonl", len(lines("tr_w20.jsonl")), seed=a.seed)
        if len(syn) < 8000: raise ValueError("insufficient nonoverlapping Crucible groups")
        if any(json.loads(r)["qid"] in all_real for r in syn): raise ValueError("Crucible/real qid collision")
        if len({group_identity(r) for r in syn[:8000]}) != 8000:
            raise ValueError("duplicate Crucible group identities")
        data["syn2"], data["syn3"] = syn[:4000], syn[4000:8000]
    real2 = sum((data["teacher2"][b] for b in BENCH), [])
    gold3 = sum((data["gold3"][b] for b in BENCH), [])
    paths = [budget_write(f"{d}/real_teacher_s2.jsonl", real2), budget_write(f"{d}/real_gold_s3.jsonl", gold3)]
    manifest = {"sample_qid_md5": sample["qid_md5"], "input_md5": inputs,
                "real_teacher_s2_md5": md5(paths[0]), "real_gold_s3_md5": md5(paths[1]),
                "base_counts": {k: {b: len(v) for b, v in data[k].items()}
                                for k in ("teacher2", "teacher3", "gold3")},
                "base_supervision": supervision_counts([real2, gold3 + sum(data["teacher3"].values(), [])], allowed, group_identity)}
    freeze_json(f"{d}/base.json", manifest)
    return {**data, "manifest": manifest, "allowed": allowed}


def build_budget(a, fill_teacher_per_bench=0):
    """Budget-only build_all branch. Fill repeats the selected stage-2 block."""
    if type(fill_teacher_per_bench) is not int or fill_teacher_per_bench < 0:
        raise ValueError("invalid budget teacher fill")
    if fill_teacher_per_bench and a.arm != "jia": raise ValueError("only jia receives budget compute fill")
    data = a._budget_data
    s2, s3 = [], []
    for b in BENCH:
        block = data["teacher2"][b]
        if not block:
            if a.cmd != "lobo" or b != a.fold:
                raise ValueError(f"{b}: empty teacher fill block")
            continue
        s2 += block + [block[i % len(block)] for i in range(fill_teacher_per_bench)]
        s3 += data["teacher3"][b] + data["gold3"][b]
    s2 += data["syn2"]; s3 += data["syn3"]
    assert_teacher_disjoint(s2, s3)
    random.Random(a.seed).shuffle(s2); random.Random(a.seed + 1).shuffle(s3)
    d = budget_dir(a) + "/planning"; os.makedirs(d, exist_ok=True)
    p2, p3 = f"{d}/s2.jsonl", f"{d}/s3.jsonl"
    _atomic_write(p2, "".join(s2)); _atomic_write(p3, "".join(s3))
    return p2, p3


def plan_budget(a, sample):
    from budget_sample import VERSION, LOBO_VERSION, freeze_json, supervision_counts
    if a.cmd == "lobo":
        frozen = validate_train_files()
        expected = {b: sorted(frozen[b]["train"]) if b != a.fold else [] for b in BENCH}
        if (a.seed != 0 or a.frac != 1.0 or a.fold not in BENCH
                or sample.get("version") != LOBO_VERSION or sample.get("fold") != a.fold
                or {b: sample["bench"][b]["qids"] for b in BENCH} != expected):
            raise ValueError("invalid LOBO real question sets")
    a = clone(a, _budget_data=prepare_budget(a, sample))
    d = budget_dir(a); frozen_path = f"{d}/plan.json"
    s1_cost = s1_flops("yi", a.seed) if a.arm == "yi" else 0.0
    def measure(fill):
        p2, p3 = build_all(a, fill)
        c2, c3 = flops(p2, 4, 4, a.seed), flops(p3, 4, 4, a.seed, 1, .5)
        if any(not math.isfinite(v) or v <= 0 for v in (c2, c3)):
            raise ValueError("invalid budget stage FLOPs")
        return p2, p3, s1_cost + c2 + c3, {"stage1": s1_cost, "stage2": c2, "stage3": c3}
    old = json.load(open(frozen_path)) if os.path.exists(frozen_path) else None
    target = (plan_budget(clone(a, arm="yi"), sample)["planned_flops"] if a.arm != "yi" else None)
    fill = old["fill_teacher_per_bench"] if old else 0
    p2, p3, value, costs = measure(fill)
    target = value if target is None else target
    if not math.isfinite(target) or target <= 0: raise ValueError("invalid yi budget FLOPs")
    if not old and a.arm == "jia":
        lo, hi = -1, 0
        while value < .95 * target:
            lo, hi = hi, max(1, 2 * hi)
            if hi > 10000000: raise ValueError("unable to bracket budget repeat fill")
            p2, p3, value, costs = measure(hi)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            trial = measure(mid)
            if trial[2] >= .95 * target: hi = mid
            else: lo = mid
        fill = hi
        p2, p3, value, costs = measure(fill)
    if a.arm == "jia" and abs(value / target - 1) > .05 + 1e-12:
        raise ValueError("budget repeat fill cannot match yi full-path FLOPs within +/-5%")
    # Freeze final paths only after an accepted search. Check existing bytes on reruns.
    p2 = budget_write(f"{d}/s2.jsonl", lines(p2)); p3 = budget_write(f"{d}/s3.jsonl", lines(p3))
    rows2, rows3 = lines(p2), lines(p3)
    steps2, steps3 = candidate_steps(len(rows2)), candidate_steps(len(rows3))
    p = {"version": VERSION, "arm": a.arm, "all_data": True, "seed": a.seed, "frac": a.frac,
         "start": all_start(a),
         "teacher": a.teacher, "teacher_config": a.teacher_config,
         "s2_groups": 16000 if a.arm == "jia" and a.cmd != "lobo" else 8000, "s3_groups": 8000, "r": .5,
         "C_yi": target, "planned_flops": value, "planned_stage_flops": costs, "ratio": value / target,
         "compute_matched": a.arm != "jiasmall", "fill_teacher_per_bench": fill,
         "s2": p2, "s3": p3, "s2_md5": md5(p2), "s3_md5": md5(p3),
         "s2_rows": len(rows2), "s3_rows": len(rows3),
         "data_freeze_md5": md5("DATA_FREEZE_STAGE23.json"),
         "sample": sample, "base": a._budget_data["manifest"], "teacher_overlap": 0,
         "group_exposures": exposure_counts(rows2, rows3),
         "supervision": supervision_counts([rows2, rows3], a._budget_data["allowed"], group_identity),
         "planned_consumed_supervision": supervision_counts([consumed_rows(p2, a.seed, steps2[-1]),
                                                             consumed_rows(p3, a.seed, steps3[-1])],
                                                            a._budget_data["allowed"], group_identity)}
    if a.cmd == "lobo":
        choice = budget_choices(a)["jia" if a.arm == "jia" else "yi"]["stage2_checkpoint"]
        selected_steps = [25, 50] if a.smoke_steps else steps2
        selected2 = selected_steps[0 if choice == "mid" else -1]
        selected3 = 50 if a.smoke_steps else steps3[-1]
        selected = [consumed_rows(p2, a.seed, selected2), consumed_rows(p3, a.seed, selected3)]
        p.update(version=LOBO_VERSION, fold=a.fold,
                 per_bench_groups=lobo_group_audit(a, rows2, rows3, a._budget_data, fill),
                 heldout_groups=0, heldout_zero_asserted=True,
                 smoke_steps=a.smoke_steps, stage2_checkpoint=choice, stage3_checkpoint="end",
                 selected_steps={"stage2": selected2, "stage3": selected3},
                 selected_supervision=supervision_counts(selected, a._budget_data["allowed"], group_identity),
                 selected_unique_supervision=lobo_unique_supervision(selected, a._budget_data["allowed"]))
    return freeze_json(frozen_path, p)


def lobo_unique_supervision(stages, allowed):
    """Content fingerprints compare unique real supervision, independently of repeats."""
    from budget_sample import canonical, digest
    sets = {k: set() for k in ("real_questions", "gold_questions", "teacher_pairs", "teacher_groups", "gold_groups")}
    for rows in stages:
        for row in map(json.loads, rows):
            if row["qid"] not in allowed: continue
            sets["real_questions"].add(row["qid"])
            kind = "teacher" if row.get("nolabel") else "gold"
            sets[kind + "_groups"].add(group_identity(row))
            if kind == "gold": sets["gold_questions"].add(row["qid"])
            if len(row["teacher"]) != len(row["docs"]): raise ValueError("unaligned teacher scores")
            for doc, score in zip(row["docs"], row["teacher"]):
                if score is not None:
                    sets["teacher_pairs"].add(canonical([row.get("teacher_name", "rr8"), row["qid"], row["query"], doc]))
    return {k: {"unique": len(v), "md5": digest(sorted(v))} for k, v in sets.items()}


def lobo_supervision_comparison(results):
    """Record exact set differences when frozen midpoint choices see different inputs."""
    required = ("selected_unique_supervision", "selected_supervision", "stage2_checkpoint")
    if any(any(key not in r or key not in r["plan"] for key in required) for r in results.values()):
        raise ValueError("LOBO result lacks selected-prefix supervision audit")
    signatures = {arm: r["selected_unique_supervision"] for arm, r in results.items()}
    for arm, r in results.items():
        if (r["plan"]["selected_unique_supervision"] != signatures[arm]
                or r["plan"]["selected_supervision"] != r["selected_supervision"]
                or r["plan"]["stage2_checkpoint"] != r["stage2_checkpoint"]):
            raise ValueError("LOBO selected supervision disagrees with frozen plan")
    reference = signatures["yi"]
    differences = {arm: {key: {"yi": reference[key], arm: value}
                         for key, value in signature.items() if value != reference[key]}
                   for arm, signature in signatures.items() if signature != reference}
    complete = set(results) == {"yi", "jia", "jiasmall"}
    return {"equal": not differences, "all_three_equal": not differences if complete else None,
            "available_arms": sorted(results),
            "missing_arms": sorted(set(("yi", "jia", "jiasmall")) - set(results)),
            "stage2_choices": {arm: r["stage2_checkpoint"] for arm, r in results.items()},
            "unique_supervision": signatures, "differences_from_yi": differences,
            "interpretation": "equal unique real supervision" if not differences else
                              "frozen selected prefixes received different unique real supervision"}


def lobo_group_audit(a, rows2, rows3, data, fill):
    """Audit final bytes by qid, including fill and Crucible collision checks."""
    frozen = json.load(open("DATA_FREEZE_STAGE23.json"))["bench"]
    identities = {b: set(frozen[b]["train"]) | set(frozen[b]["inner"]) for b in BENCH}
    heldout = identities[a.fold]
    synthetic = {json.loads(r)["qid"] for r in data["syn2"] + data["syn3"]}
    base_ids = {kind: {b: set(map(group_identity, data[kind][b])) for b in BENCH}
                for kind in ("teacher2", "teacher3", "gold3")}
    observed = {kind: {b: Counter() for b in BENCH} for kind in base_ids}
    observed_syn = {2: Counter(), 3: Counter()}
    counts = {b: {"teacher2": 0, "teacher3": 0, "gold3": 0, "fill2": 0} for b in BENCH}
    for stage, rows in ((2, rows2), (3, rows3)):
        for row in map(json.loads, rows):
            qid = row["qid"]
            if qid in heldout:
                raise ValueError("LOBO heldout qid leakage")
            owners = [b for b in BENCH if qid in identities[b]]
            if not owners:
                if qid not in synthetic: raise ValueError("LOBO unknown training qid")
                observed_syn[stage][group_identity(row)] += 1
                continue
            if len(owners) != 1 or qid not in frozen[owners[0]]["train"]:
                raise ValueError("LOBO ambiguous or inner training qid")
            kind = "teacher" if row.get("nolabel") else "gold"
            key = f"{kind}{stage}"
            if key not in counts[owners[0]]: raise ValueError("LOBO invalid stage supervision")
            if group_identity(row) not in base_ids[key][owners[0]]:
                raise ValueError("LOBO supervision escaped frozen base")
            counts[owners[0]][key] += 1
            observed[key][owners[0]][group_identity(row)] += 1
    for b in BENCH:
        counts[b]["fill2"] = fill if b != a.fold else 0
        expected = {"teacher2": 800 + fill, "teacher3": 0, "gold3": 800, "fill2": fill}
        if b == a.fold: expected = dict.fromkeys(expected, 0)
        if counts[b] != expected:
            raise ValueError(f"{b}: LOBO final counts differ from full grid recipe")
        for kind in base_ids:
            block = data[kind][b]
            if kind == "teacher2" and block:
                block = block + [block[i % len(block)] for i in range(fill)]
            if observed[kind][b] != Counter(map(group_identity, block)):
                raise ValueError(f"{b}: LOBO final row multiset differs from full grid recipe")
    for stage in (2, 3):
        if observed_syn[stage] != Counter(map(group_identity, data[f"syn{stage}"])):
            raise ValueError("LOBO Crucible allocation differs from full grid recipe")
    if any(counts[a.fold].values()): raise ValueError("LOBO heldout groups must be zero")
    return counts


def budget_stage1_actual(a):
    """Validate all_start's stage-1 chain and measured cost, never planned-as-actual."""
    if a.arm != "yi": return 0.0
    from preflight_s23 import checkpoint_recipe, training_rows
    end = all_start(a).removeprefix("/w/")
    parent = end + "_w20" if a.seed else "ckpt/w20"
    total = 0.0
    for path, world, epochs, data, start, count in (
            (parent, 4, 2, "tr_w20.jsonl", A0, 17784),
            (end, 2, 1, "tr_d3_e.jsonl", "/w/" + parent, 1500)):
        checkpoint_recipe(os.path.join(H, path), a.seed, world, epochs, "/w/" + data, start,
                          training_rows(data, count))
        expected_steps = (count // world * epochs) // 4
        if not os.path.exists(f"{path}/compute.json"):
            # Legacy seed-0 checkpoints (ckpt/w20, ckpt/d3_e) predate compute logging: verify world/steps against
            # the recorded stage-1 provenance and use the exact planned sampler FLOPs for that stage instead.
            prov = json.load(open(os.environ.get("S23_STAGE1_PROVENANCE", "stage1_provenance.json"))).get(path, {})
            if a.seed or prov.get("world") != world or prov.get("steps_completed") != expected_steps:
                raise ValueError(f"missing compute.json and no matching stage-1 provenance: {path}")
            total += flops(data, world, 4, a.seed, epochs, 0.0 if data == "tr_w20.jsonl" else 0.5)
            continue
        comp = json.load(open(f"{path}/compute.json"))
        if (comp.get("world") != world or comp.get("accum") != 4 or comp.get("steps") != expected_steps
                or not isinstance(comp.get("est_flops"), (int, float))
                or not math.isfinite(comp["est_flops"]) or comp["est_flops"] <= 0):
            raise ValueError(f"invalid actual stage-1 compute: {path}")
        total += comp["est_flops"]
    return total


def budget_question_areas(evaluation):
    """Per-qid full-inner area in points, using inner_eval's EverMem session rule."""
    from inner_eval import B, inner_questions
    from k1 import rounds_of
    frozen = json.load(open("DATA_FREEZE_STAGE23.json"))["bench"]
    result = {}
    for b in BENCH:
        qs = {q["question_id"]: q for q in inner_questions(b, frozen)}
        vals = {q: [] for q, question in qs.items() if any(r[3] for r in rounds_of(question))}
        for row in map(json.loads, lines(f"ctx_{evaluation['tag']}_inner_{b}.jsonl")):
            qid = row["qid"]
            if qid not in vals: continue
            if b == "ev":
                rounds = rounds_of(qs[qid]); gold = {r[0] for r in rounds if r[3]}
                recall = len({rounds[i][0] for i in row["keep"]} & gold) / len(gold)
            else:
                recall = row["gold_in"] / row["gold_total"]
            vals[qid].append((row["budget"], recall))
        if not vals or any(sorted(x for x, _ in v) != B for v in vals.values()):
            raise ValueError("incomplete full-inner qid areas")
        result[b] = {qid: 100 * sum(v for _, v in values) / len(B) for qid, values in vals.items()}
    area = sum(sum(v.values()) / len(v) for v in result.values()) / len(BENCH)
    if abs(area - evaluation["area"]) > .00051: raise ValueError("full-inner area disagrees with inner_score")
    return result


def budget_comparison(a):
    if a.cmd == "lobo": return  # LOBO publication belongs exclusively to lobo-summary.
    from budget_sample import paired_bootstrap, freeze_json
    paths = {arm: f"{budget_dir(clone(a, arm=arm))}/results.json" for arm in ("yi", "jia")}
    if not all(os.path.exists(p) for p in paths.values()): return
    results = {arm: json.load(open(path)) for arm, path in paths.items()}
    yi, jia = results["yi"], results["jia"]
    if (yi["plan"]["sample"] != jia["plan"]["sample"] or yi["inference"] != jia["inference"]
            or yi["plan"]["data_freeze_md5"] != jia["plan"]["data_freeze_md5"]):
        raise ValueError("incompatible budget pair")
    comparison = {"frac": a.frac, "seed": a.seed, "contrast": "yi-jia",
                  "result_md5": {arm: md5(path) for arm, path in paths.items()},
                  "planned_flops_ratio_jia_yi": jia["planned_flops"] / yi["planned_flops"],
                  "actual_flops_ratio_jia_yi": jia["actual_flops"] / yi["actual_flops"],
                  "selected_actual_flops_ratio_jia_yi": jia["selected_actual_flops"] / yi["selected_actual_flops"],
                  **paired_bootstrap(yi["inner_per_question"], jia["inner_per_question"])}
    freeze_json(f"{all_namespace(a)}/budget/comparison_f{a.frac}_s{a.seed}.json", comparison)


def lobo_comparison(a):
    """Both registered contrasts, per fold and fold-stratified heldout means."""
    from budget_sample import LOBO_VERSION, freeze_json, paired_bootstrap
    root = all_namespace(a) + "/lobo"
    summary = {"version": LOBO_VERSION, "seed": 0, "single_seed": True,
               "known_recipe_seed_sd_points": .25,
               "interval_scope": "paired qid uncertainty only; excludes training randomness",
               "claim": "无目标基准真实训练监督的迁移",
               "contrasts": {}}
    for baseline in ("jia", "jiasmall"):
        contrast = "yi-" + baseline
        folds, left, right, hashes = {}, {}, {}, {}
        for fold in BENCH:
            paths = {arm: budget_dir(clone(a, cmd="lobo", arm=arm, fold=fold)) + "/results.json"
                     for arm in ("yi", baseline)}
            if not all(os.path.exists(p) for p in paths.values()): continue
            results = {arm: json.load(open(path)) for arm, path in paths.items()}
            yi, base = results["yi"], results[baseline]
            if (any(r.get("version") != LOBO_VERSION or r.get("fold") != fold or r.get("seed") != 0
                    or r.get("arm") != arm for arm, r in results.items())
                    or yi["plan"]["sample"] != base["plan"]["sample"]
                    or yi["inference"] != base["inference"]
                    or yi["plan"]["data_freeze_md5"] != base["plan"]["data_freeze_md5"]):
                raise ValueError("incompatible LOBO pair")
            left[fold], right[fold] = yi["inner_per_question"][fold], base["inner_per_question"][fold]
            hashes[fold] = {arm: md5(path) for arm, path in paths.items()}
            per = {b: paired_bootstrap({b: yi["inner_per_question"][b]},
                                      {b: base["inner_per_question"][b]}, strata=(b,)) for b in BENCH}
            comparison = {"fold": fold, "seed": 0, "contrast": contrast,
                          "label": "C-B等" if baseline == "jia" else "C-B短",
                          "result_md5": hashes[fold], "per_bench": per,
                          "full_inner": paired_bootstrap(yi["inner_per_question"], base["inner_per_question"]),
                          "planned_flops_ratio_baseline_yi": base["planned_flops"] / yi["planned_flops"],
                          "actual_flops_ratio_baseline_yi": base["actual_flops"] / yi["actual_flops"],
                          "selected_supervision_comparison": lobo_supervision_comparison(results),
                          **per[fold]}
            freeze_json(f"{root}/comparison_{fold}_{contrast}_s0.json", comparison)
            folds[fold] = comparison
        entry = {"folds": folds, "complete": len(folds) == len(BENCH)}
        if entry["complete"]:
            aggregate = {"contrast": contrast, "seed": 0, "result_md5": hashes,
                         **paired_bootstrap(left, right)}
            aggregate["unit"] = "paired qid within heldout fold; five folds weighted equally"
            if baseline == "jia":
                aggregate["success"] = aggregate["difference"] >= 1.0 and aggregate["ci95"][0] > 0
                aggregate["success_rule"] = "mean C-B等 >= 1.0 area point and ci95 lower > 0"
            freeze_json(f"{root}/comparison_all_{contrast}_s0.json", aggregate)
            entry["five_fold"] = aggregate
        summary["contrasts"][contrast] = entry
    summary["selected_supervision_by_fold"] = {}
    for fold in BENCH:
        results = {}
        for arm in ("yi", "jia", "jiasmall"):
            path = budget_dir(clone(a, cmd="lobo", arm=arm, fold=fold, smoke_steps=0)) + "/results.json"
            if os.path.isfile(path): results[arm] = json.load(open(path))
        if "yi" in results:
            summary["selected_supervision_by_fold"][fold] = lobo_supervision_comparison(results)
    return summary


def lobo_summary(a):
    """Explicit serial publication; training workers never write comparison artifacts."""
    import fcntl
    a = clone(a, smoke_steps=0)
    root = all_namespace(a) + "/lobo"
    os.makedirs(root, exist_ok=True)
    with open(root + "/.summary.lock", "a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError("lobo-summary is already running; run summaries serially")
        summary = lobo_comparison(a)
        atomic_json(root + "/comparison_s0.json", summary)
    return summary


def lobo_smoke(a, p, choices):
    """50 steps per stage and the shared fixed 10-question INNERSEL smoke subset."""
    d = budget_dir(a)
    choice = p["stage2_checkpoint"]
    cs2 = train_all_stage(a, p, 2, all_start(a), d + "/st2")
    c2 = cs2[0 if choice == "mid" else -1]
    cs3 = train_all_stage(a, p, 3, "/w/" + c2["path"], d + "/st3")
    c3 = cs3[-1]
    if [c["step"] for c in cs2] != [25, 50] or [c["step"] for c in cs3] != [25, 50]:
        raise ValueError("LOBO smoke requires 25/50-step checkpoints")
    inference = choices["yi"]["inference"]["options"]
    evaluation = inner_score(a, c3["path"], f"lobo_{a.arm}_{a.fold}_s0_smoke50", True, inference)
    os.makedirs(d + "/inner", exist_ok=True)
    ctx_files = {}
    for b in BENCH:
        path = f"{d}/inner/ctx_{evaluation['tag']}_inner_{b}.jsonl"
        os.replace(f"ctx_{evaluation['tag']}_inner_{b}.jsonl", path)
        ctx_files[b] = {"path": path, "md5": md5(path)}
    report = {"smoke_steps": 50, "evaluation_subset": "10 fixed INNERSEL questions per benchmark",
              "plan": p, "stage2_selected": c2, "stage3_selected": c3,
              "inference": inference, "inner_report": evaluation, "inner_ctx": ctx_files}
    atomic_json(d + "/smoke_report.json", report)
    print("S23LOBO_SMOKE_DONE", d)
    return report


def lobo(a):
    return budget(clone(a, cmd="lobo", frac=1.0))


def budget(a):
    """Fixed recipes; only full-inner reporting, with no new model selection."""
    from budget_sample import FRACTIONS, VERSION, LOBO_VERSION, freeze_json, freeze_sample, supervision_counts
    is_lobo = a.cmd == "lobo"
    is_smoke = is_lobo and a.smoke_steps == 50
    if a.arm not in ("yi", "jia", "jiasmall") or a.seed not in range(5) or a.frac not in FRACTIONS:
        raise ValueError("invalid budget arm, seed or fraction")
    if is_lobo:
        if a.fold not in BENCH or a.seed != 0 or a.frac != 1.0:
            raise ValueError("LOBO requires a benchmark fold and seed 0")
    if (a.fold and not is_lobo) or ((a.smoke_steps or getattr(a, "smoke", False)) and not is_smoke):
        raise ValueError("budget/LOBO requires full all-data inner evaluation")
    if not getattr(a, "dry_run", False): stage23_gpus(a.gpus)
    mode = "lobo" if is_lobo else "budget"
    version = LOBO_VERSION if is_lobo else VERSION
    a = clone(a, all_data=True, cmd=mode, r=.5, s3_groups=8000,
              s2_groups=16000 if a.arm == "jia" and not is_lobo else 8000)
    if is_smoke: a = clone(a, smoke=True)
    root = all_namespace(a) + "/" + mode; os.makedirs(root, exist_ok=True)
    def sample_questions():
        options = {"fold": a.fold} if is_lobo else {}
        return freeze_sample(a.frac, budget_dir(a) if is_smoke else root,
                             {b: _question_ids(f"q_train_{b}.json") for b in BENCH}, **options)
    if getattr(a, "dry_run", False):
        budget_choices(a)
        print(f"{mode.title()} dry-run: CPU-only source preparation and planning; uncached FLOPs "
              "measurement uses Docker tokenization (CUDA_VISIBLE_DEVICES=, no --gpus).", flush=True)
        sample = sample_questions()
        p = plan_budget(a, sample)
        supervision = p["supervision"]
        print(json.dumps({"dry_run": True, "plan": budget_dir(a) + "/plan.json", "start": p["start"],
                          "planned_group_counts": {"stage2": p["s2_rows"], "stage3": p["s3_rows"],
                                                   "by_kind": {k: v["exposures"] for k, v in supervision["groups"].items()}},
                          "unique_scored_pairs": supervision["unique_teacher_pairs"],
                          "repeat_counts": {"scored_pairs": supervision["teacher_pairs"]["repeats"],
                                            "groups": {k: v["repeats"] for k, v in supervision["groups"].items()}},
                          "fill_teacher_per_bench": p["fill_teacher_per_bench"],
                          "planned_flops": p["planned_flops"], "C_yi": p["C_yi"], "flops_ratio": p["ratio"],
                          "planned_consumed_supervision": p["planned_consumed_supervision"],
                          **({"fold": a.fold, "per_bench_groups": p["per_bench_groups"],
                              "heldout_groups": p["heldout_groups"]} if is_lobo else {})}))
        return p
    attempt = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "arm": a.arm, "frac": a.frac, "seed": a.seed, "directory": budget_dir(a),
               **({"fold": a.fold} if is_lobo else {})}
    def record(status, **extra):
        if is_smoke: return
        with open(root + "/PROGRESS.jsonl", "a") as f:
            f.write(json.dumps({**attempt, "status": status, **extra}) + "\n")
    record("started")
    try:
        choices = budget_choices(a)
        # Shared data/base-model checks; measured stage-1 validation follows all_start.
        preflight(clone(a, arm="jia"))
        s1_actual = budget_stage1_actual(a)
        sample = sample_questions()
        p = plan_budget(a, sample); d = budget_dir(a)
        if is_smoke: return lobo_smoke(a, p, choices)
        if os.path.exists(d + "/results.json"):
            result = json.load(open(d + "/results.json"))
            if result["plan"] != p: raise ValueError("completed budget result has a different plan")
            budget_comparison(a); record("reused", results=d + "/results.json")
            return result
        frozen = choices["jia" if a.arm == "jia" else "yi"]
        start = all_start(a)
        cs2 = train_all_stage(a, p, 2, start, d + "/st2")
        c2 = cs2[0 if frozen["stage2_checkpoint"] == "mid" else -1]
        cs3 = train_all_stage(a, p, 3, "/w/" + c2["path"], d + "/st3")
        c3 = cs3[-1]
        if any(not isinstance(c.get("actual_flops"), (int, float))
               or not math.isfinite(c["actual_flops"]) or c["actual_flops"] <= 0 for c in cs2 + cs3):
            raise ValueError("invalid actual stage-2/3 compute")
        inference = choices["yi"]["inference"]["options"]
        tag = f"lobo_{a.arm}_{a.fold}_s0" if is_lobo else f"budget_{a.arm}_f{a.frac}_s{a.seed}"
        evaluation = inner_score(a, c3["path"], tag, False, inference)
        per_question = budget_question_areas(evaluation)
        # inner_score writes transient ctx files in cwd; retain budget predictions
        # beside the plan/results, without altering the shared inference helper.
        os.makedirs(d + "/inner", exist_ok=True)
        ctx_files = {}
        for b in BENCH:
            path = f"{d}/inner/ctx_{evaluation['tag']}_inner_{b}.jsonl"
            os.replace(f"ctx_{evaluation['tag']}_inner_{b}.jsonl", path)
            ctx_files[b] = {"path": path, "md5": md5(path)}
        allowed = set().union(*(set(v["qids"]) for v in sample["bench"].values()))
        selected = [consumed_rows(p["s2"], a.seed, c2["step"]), consumed_rows(p["s3"], a.seed, c3["step"])]
        executed = [consumed_rows(p["s2"], a.seed, cs2[-1]["step"]), selected[1]]
        if is_lobo and ({"stage2": c2["step"], "stage3": c3["step"]} != p["selected_steps"]
                        or lobo_unique_supervision(selected, allowed) != p["selected_unique_supervision"]):
            raise ValueError("LOBO actual selected prefix differs from frozen plan")
        result = {"version": version, "arm": a.arm, "frac": a.frac, "seed": a.seed, "plan": p,
                  "inference": inference, "stage2_checkpoint": frozen["stage2_checkpoint"], "stage3_checkpoint": "end",
                  "stage2_selected": c2, "stage3_selected": c3, "stage2_candidates": cs2, "stage3_candidates": cs3,
                  "inner_report": evaluation, "inner_ctx": ctx_files, "full_inner_area": evaluation["area"],
                  "inner_per_question": per_question,
                  "planned_flops": p["planned_flops"],
                  "actual_flops": s1_actual + cs2[-1]["actual_flops"] + c3["actual_flops"],
                  "selected_actual_flops": s1_actual + c2["actual_flops"] + c3["actual_flops"],
                  "actual_stage_flops": {"stage1": s1_actual, "stage2": cs2[-1]["actual_flops"], "stage3": c3["actual_flops"]},
                  "unique_teacher_pairs": p["supervision"]["unique_teacher_pairs"],
                  "planned_supervision": p["supervision"],
                  "actual_supervision": supervision_counts(executed, allowed, group_identity),
                  "selected_supervision": supervision_counts(selected, allowed, group_identity),
                  "actual_group_exposures": exposure_counts(*executed), "selected_group_exposures": exposure_counts(*selected)}
        if is_lobo:
            areas = {b: sum(v.values()) / len(v) for b, v in per_question.items()}
            result.update(fold=a.fold, per_bench_area=areas, heldout_area=areas[a.fold], single_seed=True,
                          selected_unique_supervision=lobo_unique_supervision(selected, allowed))
        freeze_json(d + "/results.json", result)
        budget_comparison(a)
        record("completed", results=d + "/results.json", full_inner_area=result["full_inner_area"],
               **({"heldout_area": result["heldout_area"]} if is_lobo else {}))
        print("S23LOBO_DONE" if is_lobo else "S23BUDGET_DONE", d)
        return result
    except BaseException as exc:
        record("failed", error_type=type(exc).__name__, error=str(exc))
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["plan", "run", "s1", "grid", "budget", "lobo", "lobo-summary"]); ap.add_argument("--arm")
    ap.add_argument("--fold", default=""); ap.add_argument("--all-data", action="store_true")
    ap.add_argument("--seed", type=int, choices=[0, 1, 2, 3, 4], default=0); ap.add_argument("--r", type=float, choices=[.25, .5], default=.25)
    ap.add_argument("--frac", type=float, choices=[.1, .25, 1.0], help="budget only: real supervision fraction")
    ap.add_argument("--dry-run", action="store_true", help="budget/lobo only: build sample/plan and print counts; CPU-only Docker tokenization may be required")
    ap.add_argument("--gpus", default="0,1,2,3"); ap.add_argument("--teacher"); ap.add_argument("--teacher-config")
    ap.add_argument("--gpus2", help="grid only: second disjoint set of four GPUs for parallel training/scoring")
    ap.add_argument("--s2-groups", type=int, choices=S2_LEVELS, default=8000)
    ap.add_argument("--s3-groups", type=int, choices=[4000, 8000], default=4000)
    ap.add_argument("--smoke-steps", type=int, choices=[0, 50], default=0)
    ap.add_argument("--smoke", action="store_true", help="grid only: one cell, 50 steps/stage, 10 fixed INNERSEL questions/benchmark")
    a = ap.parse_args()
    if a.cmd != "lobo-summary" and a.arm is None: ap.error("--arm is required")
    if a.teacher is None: a.teacher = "rr8+mr4" if a.cmd in ("lobo", "lobo-summary") else "rr8"
    if a.teacher_config is None: a.teacher_config = "s23/teachsel/teacher_mr4.json" if a.cmd in ("lobo", "lobo-summary") else ""
    if a.cmd != "budget" and a.seed not in (0, 1, 2): ap.error("only budget accepts seeds 3 and 4")
    if a.cmd != "budget" and a.frac is not None: ap.error("--frac applies to budget only")
    if a.cmd not in ("budget", "lobo") and a.dry_run: ap.error("--dry-run applies to budget/lobo only")
    if a.gpus2 is not None:
        if a.cmd != "grid": ap.error("--gpus2 applies to grid --all-data")
        try: gpu_sets(a)
        except ValueError as exc: ap.error(str(exc))
    if a.all_data and a.fold: ap.error("--all-data and --fold are exclusive")
    if a.smoke and a.cmd != "grid": ap.error("--smoke applies to grid --all-data")
    if a.cmd == "lobo-summary":
        if a.seed != 0 or a.fold or a.smoke_steps or a.all_data:
            ap.error("lobo-summary requires seed 0 and formal LOBO results")
        return lobo_summary(a)
    if a.cmd == "lobo":
        if a.arm not in ("yi", "jia", "jiasmall") or a.fold not in BENCH or a.seed != 0:
            ap.error("lobo requires --arm yi|jia|jiasmall, --fold beam|lme|loc|ev|ep and --seed 0")
        return lobo(a)
    if a.cmd == "budget":
        if a.frac is None: ap.error("budget requires --frac")
        if a.arm not in ("yi", "jia", "jiasmall"): ap.error("budget supports yi, jia and jiasmall")
        if a.fold or a.smoke_steps: ap.error("budget uses full all-data inner evaluation")
        return budget(a)
    if a.cmd == "grid":
        if not a.all_data: ap.error("grid requires --all-data")
        return grid(a)
    if a.all_data:
        if a.cmd != "plan": ap.error("all-data training uses grid; plan freezes one stage-2 level's four cells")
        if a.smoke_steps: ap.error("plan always uses full-path budgets; --smoke-steps applies to grid/run")
        return plan_level(a)
    if a.cmd != "s1" and a.fold not in BENCH: ap.error("choose --all-data or --fold")
    if a.teacher != "rr8" or a.teacher_config: ap.error("configurable teachers require --all-data")
    if a.arm not in ("jia", "yi", "wu", "bing", "jiap"): ap.error("unknown legacy arm")
    if a.smoke_steps and a.cmd != "run": ap.error("--smoke-steps applies to grid/run")
    return {"plan": plan, "run": run, "s1": s1}[a.cmd](a)


if __name__ == "__main__": main()
