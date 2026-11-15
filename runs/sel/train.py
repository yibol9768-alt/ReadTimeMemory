"""Train the set-conditioned selector (decoder LLM, yes/no logit at the last position) with DDP.

loss = InfoNCE over each group (1 positive + K negatives) on s = logit(yes) - logit(no)
     + w_bce * BCE(sigmoid(s), label)                      (keeps scores calibrated across groups)
     + w_kl  * KL(teacher || student) over the group, teacher = declared logits or logit(P(yes)) (redundant docs masked)
Optional focal-style weight (1 - p_pos)^gamma on the InfoNCE term (KaLM-V2).
Groups with "nolabel": true (stage-2 teacher groups, SPEC_STAGE23 v3.2) have no positive: their loss is only
w_teacher * KL(teacher || student) over the group (+ w_base KL to the starting model if --w-base > 0); InfoNCE, BCE and
--w-kl are not applied to them. Labelled groups are computed exactly as before.
Compute log (SPEC 5): non-pad training tokens, pad tokens, reference-model forward tokens, estimated FLOPs
(6 * params * train tokens + 2 * params * reference tokens) and GPU seconds, written to <out>/compute.json.

usage (inside the vLLM image, which has torch + transformers):
  torchrun --nproc_per_node 8 train.py --model /models/Qwen3-0.6B-Base --data train.jsonl --dev dev.jsonl --out ckpt/x"""
import argparse, json, math, os, random, time
import torch, torch.distributed as dist, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from teacher_s23 import logits as teacher_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--dev", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-q", type=int, default=4000); ap.add_argument("--max-d", type=int, default=800)
    ap.add_argument("--w-bce", type=float, default=0.2); ap.add_argument("--w-kl", type=float, default=0.0)
    ap.add_argument("--w-base", type=float, default=0.0, help="KL to the frozen starting model over each group: keeps what the base already does well")
    ap.add_argument("--save-at", default="", help="comma list of steps to save checkpoints <out>/s<step>")
    ap.add_argument("--gamma", type=float, default=0.0); ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--group", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=200); ap.add_argument("--shuffle-docs", type=int, default=1)
    ap.add_argument("--w-teacher", type=float, default=1.0, help="weight of KL(teacher||student) on nolabel groups")
    ap.add_argument("--save-every", type=int, default=0, help="save <out>/s<step> every N optimizer steps (SPEC: 200)")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test only: cap optimizer steps (0 = full stage)")
    a = ap.parse_args()

    dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size()
    lrk = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(lrk); dev = torch.device("cuda", lrk)
    torch.manual_seed(a.seed); random.seed(a.seed + rank)
    tok = AutoTokenizer.from_pretrained(a.model)
    yes_id, no_id = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.config.use_cache = False

    class Scorer(torch.nn.Module):
        def __init__(self, lm):
            super().__init__(); self.lm = lm

        def forward(self, ids, att, last):
            hs = self.lm.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
            hl = hs[torch.arange(ids.size(0), device=ids.device), last]
            w = self.lm.get_output_embeddings().weight[[yes_id, no_id]]
            lg = hl.float() @ w.float().t()
            return lg[:, 0] - lg[:, 1]

    ddp = torch.nn.parallel.DistributedDataParallel(Scorer(model), device_ids=[lrk], find_unused_parameters=False)
    base = None
    if a.w_base > 0:
        base = Scorer(AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev)).eval()
        for p in base.parameters(): p.requires_grad_(False)
    save_at = {int(x) for x in a.save_at.split(",") if x}

    def save(path, saved_step):
        os.makedirs(path, exist_ok=True); sd = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in model.state_dict().items()}
        model.save_pretrained(path, state_dict=sd, safe_serialization=True); tok.save_pretrained(path)
        # Write last: selection must never accept a directory merely because it exists.
        json.dump({**vars(a), "checkpoint_step": saved_step, "world": world, "steps_total": steps_total},
                  open(os.path.join(path, "train_args.json"), "w"), indent=1)

    rows = [json.loads(l) for l in open(a.data)]
    random.Random(a.seed).shuffle(rows)
    mine = rows[rank::world]
    devrows = [json.loads(l) for l in open(a.dev)][:400] if a.dev else []
    sfx_ids = tok("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)["input_ids"]

    trunc = [0]

    def encode(r):
        q = tok(r["query"], add_special_tokens=False)["input_ids"]
        if len(q) > a.max_q: q = q[:200] + q[-(a.max_q - 200):]; trunc[0] += 1   # should not happen (fmt bounds lengths)
        docs = r["docs"][:a.group]; seqs = []
        for d in docs:
            body = d[: -len("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")] if d.endswith("</think>\n\n") else d
            di = tok(body, add_special_tokens=False)["input_ids"][: a.max_d]
            seqs.append(q + di + sfx_ids)
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), tok.pad_token_id or 0, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for k, s in enumerate(seqs): ids[k, :len(s)] = torch.tensor(s); att[k, :len(s)] = 1
        return ids.to(dev), att.to(dev), torch.tensor([len(s) - 1 for s in seqs], device=dev)

    def scores(m, ids, att, last):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m(ids, att, last)

    cnt = {"train_tok": 0, "pad_tok": 0, "ref_tok": 0, "groups": 0, "nolabel_groups": 0}

    def kl_teacher(s, t, scale="probability"):
        idx = torch.tensor([k for k, v in enumerate(t) if v is not None], device=dev)
        tl = torch.tensor(teacher_logits([t[k] for k in idx.tolist()], scale), device=dev)
        return F.kl_div(F.log_softmax(s[idx], 0), F.log_softmax(tl, 0), log_target=True, reduction="sum")

    def group_loss(r, m):
        docs = list(range(min(len(r["docs"]), a.group)))
        if a.shuffle_docs: random.shuffle(docs)
        kinds = r.get("kinds") or ["t"] * len(r["docs"])
        rr = {**r, "docs": [r["docs"][k] for k in docs], "kinds": [kinds[k] for k in docs],
              "teacher": [r["teacher"][k] for k in docs]}
        ids, att, last = encode(rr)
        n_real = int(att.sum().item()); cnt["train_tok"] += n_real; cnt["pad_tok"] += att.numel() - n_real; cnt["groups"] += 1
        s = scores(m, ids, att, last)
        if r.get("nolabel"):
            cnt["nolabel_groups"] += 1
            if sum(x is not None for x in rr["teacher"]) < 2: raise ValueError("nolabel group with fewer than 2 teacher scores")
            loss = a.w_teacher * kl_teacher(s, rr["teacher"], r.get("teacher_format", "probability"))
            if base is not None:
                with torch.no_grad(): sb = scores(base, ids, att, last)
                cnt["ref_tok"] += n_real
                loss = loss + a.w_base * F.kl_div(F.log_softmax(s, 0), F.log_softmax(sb.float(), 0), log_target=True, reduction="sum")
            return loss, 0
        lab = docs.index(0)
        logp = F.log_softmax(s / a.tau, 0)
        nce = -logp[lab]
        if a.gamma > 0: nce = nce * (1 - logp[lab].exp().detach()) ** a.gamma
        y = torch.zeros_like(s); y[lab] = 1
        loss = nce + a.w_bce * F.binary_cross_entropy_with_logits(s, y)
        if base is not None:
            with torch.no_grad(): sb = scores(base, ids, att, last)
            cnt["ref_tok"] += n_real
            loss = loss + a.w_base * F.kl_div(F.log_softmax(s, 0), F.log_softmax(sb.float(), 0), log_target=True, reduction="sum")
        if a.w_kl > 0:
            t = rr["teacher"]; mask = [x is not None and kd != "red" for x, kd in zip(t, rr["kinds"])]
            if sum(mask) >= 2:
                idx = torch.tensor([k for k, v in enumerate(mask) if v], device=dev)
                tl = torch.tensor(teacher_logits([t[k] for k in idx.tolist()], r.get("teacher_format", "probability")), device=dev)
                loss = loss + a.w_kl * F.kl_div(F.log_softmax(s[idx], 0), F.log_softmax(tl, 0), log_target=True, reduction="sum")
        return loss, int(s.argmax().item() == lab)

    steps_total = int((len(rows) // world) * a.epochs) // a.accum   # same on every rank (audit: rank mismatch hangs DDP)
    if a.max_steps: steps_total = min(steps_total, a.max_steps)
    opt = torch.optim.AdamW(ddp.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, int(0.05 * steps_total))) *
                                              0.5 * (1 + math.cos(math.pi * min(1, s / max(1, steps_total)))))

    eval_s = [0.0]

    def evaluate():
        te = time.time(); ddp.eval(); hit = n = 0
        with torch.no_grad():
            for r in devrows[rank::world]:
                ids, att, last = encode(r); s = scores(ddp, ids, att, last); hit += int(s.argmax().item() == 0); n += 1
        t = torch.tensor([hit, n], device=dev); dist.all_reduce(t); ddp.train(); eval_s[0] += time.time() - te
        return (t[0] / max(1, t[1])).item()

    if devrows:
        acc0 = evaluate()
        if rank == 0: print(f"step 0 dev_top1 {acc0:.3f}", flush=True)
    NPARAM = sum(p.numel() for p in model.parameters())
    eval_pre = eval_s[0]; ddp.train(); step = 0; t0 = time.time(); run_l = run_a = 0.0; it = 0; eval_s[0] = 0.0   # step-0 eval kept in eval_pre
    order = []
    while len(order) < steps_total * a.accum: order += random.sample(range(len(mine)), len(mine))
    for k in range(steps_total * a.accum):
        r = mine[order[k]]
        ctx = ddp.no_sync() if (k + 1) % a.accum else torch.enable_grad()
        with ctx:
            loss, acc = group_loss(r, ddp)
            (loss / a.accum).backward()
        run_l += loss.item(); run_a += acc; it += 1
        if (k + 1) % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            cs = torch.tensor([cnt["train_tok"], cnt["pad_tok"], cnt["ref_tok"], cnt["groups"]], dtype=torch.float64, device=dev); dist.all_reduce(cs)
            if rank == 0:   # cumulative compute after every optimizer step (SPEC 2), so a selected checkpoint's FLOPs are known
                if step == 1: os.makedirs(a.out, exist_ok=True); clog = open(os.path.join(a.out, "compute_steps.jsonl"), "w")
                clog.write(json.dumps({"step": step, "train_tokens": int(cs[0]), "pad_tokens": int(cs[1]), "ref_forward_tokens": int(cs[2]), "groups": int(cs[3]),
                                       "est_flops": 6 * NPARAM * int(cs[0]) + 2 * NPARAM * int(cs[2]), "train_wall_seconds": round(time.time() - t0 - eval_s[0], 2),
                                       "train_gpu_seconds": round((time.time() - t0 - eval_s[0]) * world, 2), "eval_wall_seconds": round(eval_pre + eval_s[0], 2)}) + "\n"); clog.flush()
            if rank == 0 and step % 20 == 0:
                print(f"step {step}/{steps_total} loss {run_l / it:.4f} train_top1 {run_a / it:.3f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
                run_l = run_a = 0.0; it = 0
            if rank == 0 and (step in save_at or (a.save_every and step % a.save_every == 0)): save(os.path.join(a.out, f"s{step}"), step); print("CKPT", step, flush=True)
            if devrows and step % a.eval_every == 0:
                acc = evaluate()
                if rank == 0: print(f"step {step} dev_top1 {acc:.3f}", flush=True)
    if devrows:
        acc = evaluate()
        if rank == 0: print(f"final dev_top1 {acc:.3f}", flush=True)
    c = torch.tensor([cnt[k] for k in ("train_tok", "pad_tok", "ref_tok", "groups", "nolabel_groups")], dtype=torch.float64, device=dev)
    dist.all_reduce(c); gpu_s = torch.tensor([time.time() - t0 - eval_s[0]], dtype=torch.float64, device=dev); dist.all_reduce(gpu_s)
    nparam = sum(p.numel() for p in model.parameters())
    comp = {"params": nparam, "train_tokens": int(c[0]), "pad_tokens": int(c[1]), "ref_forward_tokens": int(c[2]), "groups": int(c[3]),
            "nolabel_groups": int(c[4]), "est_flops": 6 * nparam * int(c[0]) + 2 * nparam * int(c[2]), "gpu_seconds_excl_eval": float(gpu_s[0]), "eval_seconds_rank0": round(eval_pre + eval_s[0], 2),
            "steps": step, "world": world, "accum": a.accum, "note": "FLOPs exclude padding and gradient-checkpointing recompute (SPEC_STAGE23 v3.2 5)"}
    if rank == 0:
        os.makedirs(a.out, exist_ok=True); json.dump(comp, open(os.path.join(a.out, "compute.json"), "w"), indent=1); print("COMPUTE", json.dumps(comp), flush=True)
        save(a.out, step)
        print("SAVED", a.out, flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
