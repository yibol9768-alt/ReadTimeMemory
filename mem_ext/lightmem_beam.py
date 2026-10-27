"""LightMem (ICLR 2026, arXiv 2510.18866) on BEAM-100K dev, run in its own env (~/v3venvs/lightmem).

Faithful to the repo's LongMemEval runner (experiments/longmemeval/run_lightmem_gpt.py): LLMLingua-2 pre-compression
and topic segmentation, messages_use=user_only, metadata generation and text summary, all-MiniLM-L6-v2 embeddings,
qdrant, update=offline config (the runner itself does not call the offline update), retrieve(limit=20).
Changes: memory manager model = gpt-6-luna on our Pro proxy (OpenAI-compatible chat completions); CPU instead of
CUDA; BEAM dates "March-15-2024" -> "2024/03/15 (Fri) 00:00". Answering and judging are done by mem/beam.py with the
same luna reader and the official BEAM judge as every other condition; this script only writes the retrieved context.

Sensitivity variants (one change each): --messages-use hybrid (index user and assistant turns), --limit N
(retrieve N entries instead of 20). Each variant writes to its own condition folder (lightmem_<tag>).

Usage: ~/v3venvs/lightmem/bin/python mem_ext/lightmem_beam.py [--convs 0,10,14,16,17,18] [--messages-use hybrid] [--limit 60]
"""
import argparse, ast, datetime, json, os, re, shutil, time

V3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(V3, "data", "beam", "100K_rows.json")
OUT = os.path.join(V3, "runs", "beam", "pro6", "ext", "lightmem")
STORE = os.path.expanduser("~/v3venvs/lm_store")
ABILITIES = ["abstention", "contradiction_resolution", "event_ordering", "information_extraction",
             "instruction_following", "knowledge_update", "multi_session_reasoning", "preference_following",
             "summarization", "temporal_reasoning"]


def key():
    line = open(os.path.expanduser("~/.local/share/codex-workflow/llm-proxy-private/config-pro.yaml")).read().splitlines()[12]
    return re.sub(r'^\s*-\s*"?([^"]+?)"?\s*$', r"\1", line)


def stamp(anchor):
    d = datetime.datetime.strptime(anchor, "%B-%d-%Y")
    return d.strftime("%Y/%m/%d (%a) 00:00")


def load_lightmem(name, messages_use="user_only", wipe=True):
    from lightmem.memory.lightmem import LightMemory
    path = os.path.join(STORE, name)
    if wipe:
        shutil.rmtree(path, ignore_errors=True)
    return LightMemory.from_config({
        "pre_compress": True,
        "pre_compressor": {"model_name": "llmlingua-2", "configs": {"llmlingua_config": {
            "model_name": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            "device_map": "cpu", "use_llmlingua2": True}}},
        "topic_segment": True, "precomp_topic_shared": True,
        "topic_segmenter": {"model_name": "llmlingua-2"},
        "messages_use": messages_use, "metadata_generate": True, "text_summary": True,
        "memory_manager": {"model_name": "openai", "configs": {
            "model": "gpt-6-luna", "api_key": key(), "max_tokens": 16000,
            "openai_base_url": "http://127.0.0.1:18318/v1"}},
        "extract_threshold": 0.1, "index_strategy": "embedding",
        "text_embedder": {"model_name": "huggingface", "configs": {
            "model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_dims": 384, "model_kwargs": {"device": "cpu"}}},
        "retrieve_strategy": "embedding",
        "embedding_retriever": {"model_name": "qdrant", "configs": {
            "collection_name": name, "embedding_model_dims": 384, "path": path}},
        "update": "offline",
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", default="0,10,14,16,17,18")
    ap.add_argument("--messages-use", default="user_only")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--also-limits", default="", help="extra retrieval sizes written from the same store, e.g. 60")
    args = ap.parse_args()
    tag = "".join([f"_{args.messages_use}" if args.messages_use != "user_only" else "",
                   f"_k{args.limit}" if args.limit != 20 else ""])
    global OUT
    OUT = OUT + tag
    os.makedirs(OUT, exist_ok=True)
    rows = [x["row"] for x in json.load(open(DATA))["rows"]]
    path = os.path.join(OUT, "ctx.jsonl")
    done = {r["qid"] for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    for ci in map(int, args.convs.split(",")):
        r = rows[ci]
        probing = ast.literal_eval(r["probing_questions"])
        qs = [(f"b{ci}-{ab}-{j}", q["question"]) for ab in ABILITIES for j, q in enumerate(probing.get(ab, []))]
        if all(q in done for q, _ in qs):
            continue
        t0 = time.time()
        # LightMem's init recreates the collection, so a store cannot be reopened: always rebuild, and write every
        # retrieval size needed from the same store (--also-limits)
        lm = load_lightmem(f"beam100k_{ci}{tag}", args.messages_use)
        chunks = r["chat"]
        for k, chunk in enumerate(chunks):
            ts = stamp(next(t["time_anchor"] for t in chunk if t.get("time_anchor")))
            msgs = [{"role": t["role"], "content": re.sub(r"\s*->->\s*\d+,\d+\s*$", "", t["content"]).strip()}
                    for t in chunk]
            while msgs and msgs[0]["role"] != "user":
                msgs.pop(0)
            pairs = [msgs[i:i + 2] for i in range(0, len(msgs) - 1, 2)
                     if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant"]
            for i, pair in enumerate(pairs):
                for m in pair:
                    m["time_stamp"] = ts
                last = k == len(chunks) - 1 and i == len(pairs) - 1
                lm.add_memory(messages=pair, force_segment=last, force_extract=last)
        build = time.time() - t0
        with open(path, "a") as f:
            for qid, q in qs:
                if qid in done:
                    continue
                mems = lm.retrieve(q, limit=args.limit)
                f.write(json.dumps({"qid": qid, "context": "\n".join(mems), "n": len(mems), "build_s": round(build)},
                                   ensure_ascii=False) + "\n")
        for extra in [int(x) for x in args.also_limits.split(",") if x]:
            d = OUT + f"_k{extra}"
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "ctx.jsonl"), "a") as f:
                for qid, q in qs:
                    mems = lm.retrieve(q, limit=extra)
                    f.write(json.dumps({"qid": qid, "context": "\n".join(mems), "n": len(mems), "build_s": round(build)},
                                       ensure_ascii=False) + "\n")
        print("lightmem conv", ci, "built in", round(build), "s", flush=True)


if __name__ == "__main__":
    main()
