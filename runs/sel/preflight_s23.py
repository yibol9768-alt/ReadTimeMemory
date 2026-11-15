"""Read-only, stdlib-only preflight. Never starts a process or loads a model."""
import hashlib
import json
import os

from inner_eval import validate_selection, inner_questions, smoke_ids
from k1 import rounds_of
from teacher_s23 import teacher_maps

BENCH = ("beam", "lme", "loc", "ev", "ep")
# Canonical train/inner identities from DATA_FREEZE_STAGE23 (2026-11-02).
SPLIT_IDENTITY_SHA256 = "e6ed5a7ef31e374fdb59a3a040166db70f83bf7891ce36643950c6657dbd5ddd"


def digest(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def split_hash(frozen):
    canonical = {b: {k: sorted(v[k]) for k in ("train", "inner")} for b, v in frozen.items()}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def host_path(path, home, models):
    if path.startswith("/w/"): return os.path.join(home, path[3:])
    if path.startswith("/models/"): return os.path.join(models, path[8:])
    return path


def model_assets(path):
    config = json.load(open(f"{path}/config.json"))
    if config.get("model_type") != "qwen3": raise ValueError(f"unexpected checkpoint architecture: {path}")
    index = f"{path}/model.safetensors.index.json"
    shards = (set(json.load(open(index))["weight_map"].values()) if os.path.isfile(index)
              else {"model.safetensors"})
    if not shards or any(not os.path.isfile(f"{path}/{s}") or os.path.getsize(f"{path}/{s}") == 0 for s in shards):
        raise ValueError(f"missing/empty checkpoint weights: {path}")
    for name in ("tokenizer_config.json", "tokenizer.json"):
        json.load(open(f"{path}/{name}"))


def training_rows(path, count=None):
    n = 0
    with open(path) as f:
        for line in f:
            row = json.loads(line); docs = row.get("docs", [])
            if (not isinstance(row.get("query"), str) or not 5 <= len(docs) <= 16
                    or not all(isinstance(d, str) for d in docs) or row.get("label") != 0
                    or row.get("nolabel") or len(row.get("teacher", [])) != len(docs)):
                raise ValueError(f"invalid stage-1 group: {path}:{n + 1}")
            n += 1
    if not n or (count is not None and n != count): raise ValueError(f"wrong stage-1 row count: {path}: {n}, expected {count}")
    return n


def checkpoint_recipe(path, seed, world, epochs, data, start, rows):
    """Require complete recorded provenance; never infer missing world/steps."""
    model_assets(path)
    args = json.load(open(f"{path}/train_args.json"))
    # Old root-owned stage-1 checkpoints lack world/steps; a trusted sidecar (env S23_STAGE1_PROVENANCE, keyed "ckpt/<name>")
    # may supply ONLY missing fields. Values still go through the same equality checks below, so a wrong sidecar fails.
    prov_file = os.environ.get("S23_STAGE1_PROVENANCE", "")
    if prov_file:
        key = "ckpt/" + os.path.basename(os.path.normpath(path)); prov = json.load(open(prov_file)).get(key)
        if prov is not None:
            fill = dict(world=prov.get("world"), checkpoint_step=prov.get("steps_completed"), steps_total=prov.get("steps_completed"))
            if "w_base" in prov: fill["w_base"] = prov["w_base"]   # only for checkpoints trained before train.py had --w-base
            for k, v in fill.items():
                if v is None: raise ValueError(f"stage-1 provenance missing {k}: {key}")
                if k in args and args[k] != v: raise ValueError(f"stage-1 provenance conflict: {key}: {k} {args[k]!r} vs {v!r}")
                args.setdefault(k, v)
    expected = dict(seed=seed, world=world, epochs=epochs, accum=4, lr=1e-5,
                    max_q=4000, max_d=1100, group=16, tau=1.0, gamma=0,
                    shuffle_docs=1, w_bce=.2, w_kl=0.0, w_base=.5 if world == 2 else 0.0)
    steps = int((rows // world) * epochs) // 4
    expected.update(checkpoint_step=steps, steps_total=steps)
    for key, value in expected.items():
        if args.get(key) != value: raise ValueError(f"stage-1 config mismatch: {path}: {key}={args.get(key)!r}, expected {value!r}")
    # Compare container-relative identities; a prepared copy may be mounted elsewhere on the host.
    if args.get("data") != data or args.get("model") != start:
        raise ValueError(f"stage-1 data/start mismatch: {path}")


def question_rounds(q):
    for key in ("question_id", "question", "question_date", "question_type"):
        if not isinstance(q.get(key), str): raise ValueError(f"invalid question field {key}")
    sessions, dates = q["haystack_sessions"], q["haystack_dates"]
    if len(sessions) != len(dates) or not sessions: raise ValueError("invalid session/date alignment")
    for date, session in zip(dates, sessions):
        if not isinstance(date, str): raise ValueError("invalid session date")
        for turn in session:
            if turn.get("role") not in ("user", "assistant", "system") or not isinstance(turn.get("content"), str):
                raise ValueError("invalid conversation turn")
    rounds = rounds_of(q)
    if not rounds: raise ValueError("empty question history")
    return rounds


def check_pool(q, manifest, training=False):
    qid = q["question_id"]; path = f"pool_s23/{qid}.json"
    if manifest.get(qid) != digest(path): raise ValueError(f"pool hash mismatch: {qid}")
    pool = json.load(open(path)); rounds = question_rounds(q); fused = pool["fused"]
    if (len(fused) != min(600, len(rounds)) or len(fused) != len(set(fused))
            or any(type(i) is not int or not 0 <= i < len(rounds) for i in fused)):
        raise ValueError(f"invalid top-600 pool: {qid}")
    if training:
        if pool.get("gold") != [r[3] for r in rounds] or len(pool.get("stage", [])) != len(rounds):
            raise ValueError(f"pool gold/stage alignment mismatch: {qid}")
        # Gold-group generation always uses RR8 hard negatives, even with another distillation teacher.
        rr8 = pool.get("rr8", {})
        from teacher_s23 import logits
        scores = [rr8.get(str(i)) for i in fused[:300]]
        if any(v is None for v in scores): raise ValueError(f"missing training RR8 scores: {qid}")
        logits(scores)
    return pool


def data_preflight(a):
    frozen = json.load(open("DATA_FREEZE_STAGE23.json"))["bench"]
    if set(frozen) != set(BENCH) or split_hash(frozen) != SPLIT_IDENTITY_SHA256:
        raise ValueError("frozen train/inner identity hash mismatch")
    selection = json.load(open("INNERSEL_STAGE23.json"))
    selected = validate_selection(frozen, selection)
    manifest = json.load(open("POOL_S23_MD5.json"))
    counts = dict(selection=0, full=0, smoke=0)
    for b in BENCH:
        train = json.load(open(f"q_train_{b}.json")); ids = [q["question_id"] for q in train]
        if (len(ids) != len(set(ids)) or set(ids) != set(frozen[b]["train"])
                or set(ids) & set(frozen[b]["inner"])):
            raise ValueError(f"{b}: train identities differ from frozen train-minus-inner")
        full = inner_questions(b, frozen)
        subset = inner_questions(b, selected, True)
        by_id = {q["question_id"]: q for q in full}
        if any(q != by_id[q["question_id"]] for q in subset):
            raise ValueError(f"{b}: INNERSEL question contents differ from full inner")
        small = smoke_ids(selection[b], b)
        for label, questions in (("full", full), ("smoke", [by_id[i] for i in small])):
            if not any(any(r[3] for r in question_rounds(q)) for q in questions):
                raise ValueError(f"{b}: {label} inner has no gold questions")
        usable = gold_usable = 0
        for q in train:
            pool = check_pool(q, manifest, True)
            maps = teacher_maps(pool, q["question_id"], getattr(a, "teacher", "rr8"), getattr(a, "teacher_config", ""))
            usable += sum(all(i in m for m in maps) for i in pool["fused"][:300]) >= 28
            negatives = [i for i in pool["fused"] if not pool["gold"][i]]
            skipped = min(10, sum(str(i) in pool["rr8"] for i in negatives))
            gold_usable += any(pool["gold"][i] for i in pool["fused"]) and len(negatives) - skipped >= 4
        if not usable: raise ValueError(f"{b}: no usable teacher questions")
        if not gold_usable: raise ValueError(f"{b}: no usable gold questions after hard-skip 10")
        for q in full:
            pool = check_pool(q, manifest); n = len(pool["fused"][:600]); qid = q["question_id"]
            counts["full"] += n
            if qid in selected[b]["inner"]: counts["selection"] += n
            if qid in small: counts["smoke"] += n
    return counts


def preflight(a, home, models, stage1=False):
    """Validate the whole downstream data path before any training, including s1."""
    counts = data_preflight(a)
    n_w20 = training_rows("tr_w20.jsonl", 17784)
    n_d3 = training_rows("tr_d3_e.jsonl", 1500)
    base = "/models/qwen/Qwen3-Reranker-0.6B"
    model_assets(host_path(base, home, models))
    if stage1:
        if a.arm == "bing": training_rows("trx.jsonl")
    elif a.arm == "yi":
        end = f"/w/ckpt/s23_s1_yi_s{a.seed}" if a.seed else "/w/ckpt/d3_e"
        start = end + "_w20" if a.seed else "/w/ckpt/w20"
        checkpoint_recipe(host_path(start, home, models), a.seed, 4, 2, "/w/tr_w20.jsonl", base, n_w20)
        checkpoint_recipe(host_path(end, home, models), a.seed, 2, 1, "/w/tr_d3_e.jsonl", start, n_d3)
    return counts
