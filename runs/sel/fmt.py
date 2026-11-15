"""Shared prompt format for the set-conditioned evidence selector (SetSel).

Same surface format as Qwen3-Reranker (query string + document string, yes/no at the last position), so
(1) Qwen3-Reranker-0.6B is a drop-in initialisation and baseline, and (2) a trained checkpoint can be served by vLLM's
/v1/rerank with the usual Qwen3 reranker overrides. The query adds a <Selected> block (rounds already chosen), so the
score means "adds needed information that is not yet covered".

Review fixes (10-31):
- no gold question type anywhere (was leakage); one generic instruction for every question,
- every round carries its context: the previous user message of the same session (content-free follow-ups like
  "hmm, which is better?" are otherwise unjudgeable), both in documents and in the selected block,
- the selected block keeps up to 40 items (16k budgets select 30+), each a short user snippet + context.
Lengths are bounded here so the trainer never needs to truncate (train/infer parity)."""
import os, re

PFX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
       "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SFX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

INST_V1 = ("Given a question a user asks about their own past chats (asked on {qdate}) and the chat rounds already selected "
           "as evidence, judge whether this chat round adds information needed to answer the question that the selected "
           "rounds do not already provide. If the question asks for an ordered list, a summary or several facts, a round "
           "that covers a part not yet covered is needed; a round repeating an already covered part is not")
# v2 (11-02, stage-1 failure study): neutral about who spoke. v1 ("their own past chats") taught "evidence = the asker's own
# words"; on group chats E0 put the self-named asker's messages first (EverMem P_* questions). v1 stays the default so that
# every existing model and result reproduces; v2 is chosen with FMT_INST=2 and must be the same at training and inference.
INST_V2 = ("Given a question about past conversations (asked on {qdate}) and the rounds already selected as evidence, judge "
           "whether this round adds information needed to answer the question that the selected rounds do not already "
           "provide. The conversations may be one-on-one chats with an assistant or group chats among several people; the "
           "person asking may or may not be one of the speakers, and the needed information may come from anyone. If the "
           "question asks for an ordered list, a summary or several facts, a round that covers a part not yet covered is "
           "needed; a round repeating an already covered part is not")
INST_VERSION = os.environ.get("FMT_INST", "1")
assert INST_VERSION in ("1", "2"), f"FMT_INST must be 1 or 2, got {INST_VERSION!r}"
INST = INST_V2 if INST_VERSION == "2" else INST_V1


MARK = re.compile(r"\s*->->\s*\d+\s*,\s*(?:\d+(?:\s*,\s*\d+)*|N/A)?")


def clean(t): return re.sub(r"\s+", " ", MARK.sub("", t or "")).strip()


def snippet(user, prev="", n=260):   # 160 cut the key value of about 10% of world rounds (checked 11-01)
    u = clean(user)
    if prev and len(u) < 160: return f"(re: {clean(prev)[:90]}) {u[:n]}"
    return u[:n]


def query_str(question, qdate, selected, max_sel=40):
    """selected: list of (date, user_text, prev_user_text) in selection order."""
    sel = selected[-max_sel:]
    if sel:
        s = "\n".join(f"[{k + 1}] ({d}) {snippet(u, p)}" for k, (d, u, p) in enumerate(sel))
        if len(selected) > max_sel: s = f"({len(selected) - max_sel} earlier selections omitted)\n" + s
    else:
        s = "(none yet)"
    return PFX + f"<Instruct>: {INST.format(qdate=qdate)}\n<Query>: {question}\n<Selected>:\n{s}\n"


def doc_str(date, user, assistant, prev="", max_chars=3000, *, neighbour_context=False, neighbours=()):   # 3000 = same as the 8B baseline (k1._rr_scores)
    if os.environ.get("MASK_DATE"): date = "-"   # probe only: how much of a model's ranking rests on the candidate date
    ctx = f"[follow-up to: {clean(prev)[:200]}]\n" if prev else ""
    body = f"User: {user}\nAssistant: {assistant}"
    if neighbour_context:
        adjacent = "\n".join(f"({d}) {clean(t)[:700]}" for d, t in neighbours)
        body = body[:max_chars] + ("\n[Adjacent rounds]\n" + adjacent if adjacent else "")
    return f"<Document>: ({date}) {ctx}{body if neighbour_context else body[:max_chars]}" + SFX


def split_round(text):
    """k1 round text 'User: ...\\nAssistant: ...' -> (user, assistant)."""
    text = MARK.sub("", text)   # BEAM plan markers ('->-> 2,10', ~76% of rounds) must never reach the selector
    m = re.match(r"User: (.*?)(?:\nAssistant: (.*))?$", text, re.S)
    return (m.group(1), m.group(2) or "") if m else (text, "")


def round_prev(q):
    """previous user message in the same session for every k1 round (aligned with k1.rounds_of)."""
    out = []
    for sess in q["haystack_sessions"]:
        i, prev = 0, ""
        while i < len(sess):
            first = sess[i]
            if sess[i]["role"] == "user" and i + 1 < len(sess) and sess[i + 1]["role"] == "assistant": i += 1
            i += 1
            short = first["role"] == "user" and len((first["content"] or "").split()) < 30
            out.append(prev if short else "")   # audit fix: context only for short follow-ups
            if first["role"] == "user": prev = first["content"]
    return out
