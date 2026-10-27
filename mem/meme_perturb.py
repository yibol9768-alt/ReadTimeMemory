"""Robustness variant of MEME: rewrite every user turn that states a dependency in MEME's fixed template
("... this is determined by X; if that changes, this would change", "If we change X, Y will be Z") into a natural,
implicit phrasing (DeepSeek), keeping every name and value verbatim. A rewrite that drops a value is retried, then the
original turn is kept and counted.

Usage: python3 -m mem.meme_perturb --data IN.json --ids a,b,... --out OUT.json
"""
import argparse, json, re
from concurrent.futures import ThreadPoolExecutor
from . import llm

PAT = re.compile(r"determined by|if (that|this|it) changes|would change|^\s*if (we|i|my|our|the)\b.*\b(change|move|switch)",
                 re.I | re.S)

SYS = ("Rewrite the user's chat message so it sounds like a natural, casual remark, keeping exactly the same meaning. "
       "The message either (a) gives a current value and says it is tied to something else, or (b) says what a value "
       "would become if something else changed. Keep the tie and, for (b), keep it clearly hypothetical. Say it the way "
       "people actually talk, without the words 'determined', 'depends', 'dependency' or the phrase 'if that changes'.\n"
       "Example (a): 'Our CI config is a.toml — this is determined by our infrastructure (we deploy on Foo Cloud); if "
       "that changes, this would change.' -> 'Our CI config is a.toml, which is really just because we deploy on Foo "
       "Cloud.'\nExample (b): 'If we change deploy targets, the CI config will be b.yml.' -> 'Should we ever leave our "
       "current deploy target, we'd move the CI config over to b.yml.'\n"
       "Copy every proper name, value, path, command and number exactly as written. Keep any question at the end. "
       "Output only the rewritten message.")


def values_in(text):
    """Tokens that must survive: quoted/capitalised names, paths, commands, numbers."""
    toks = {t.strip(".,;:'") for t in re.findall(r"[A-Za-z0-9_\-./:']+", text)}
    return {t for t in toks if (len(t) > 2 or t.isdigit()) and re.search(r"[A-Z0-9/_]|\w\.\w", t) and t.lower() not in
            {"our", "the", "any", "this", "that", "if", "now", "update", "do", "have", "how", "what"}}


TEMPLATE = re.compile(r"determined by|if (that|this|it) changes|this would change", re.I)


def rewrite(turn):
    need = values_in(turn)
    for g in range(1, 4):
        import os
        os.environ["V3_GEN"] = str(g)
        out = (llm.call(SYS, turn, effort="low", max_tokens=1200, tag="meme_perturb") or "").strip()
        if out and not TEMPLATE.search(out) and all(v in out for v in need):
            return out, True
    print("FAIL", [v for v in need if v not in out], "|", out[:200].replace("\n", " "), flush=True)
    return turn, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--ids"); ap.add_argument("--out")
    a = ap.parse_args()
    data = [e for e in json.load(open(a.data)) if e["episode_id"] in a.ids.split(",")]
    jobs = [(ei, si, ti) for ei, e in enumerate(data) for si, s in enumerate(e["sessions"])
            for ti, t in enumerate(s["conversation"]) if t["role"] == "user" and PAT.search(t["content"])]
    with ThreadPoolExecutor(2) as ex:
        res = list(ex.map(lambda j: rewrite(data[j[0]]["sessions"][j[1]]["conversation"][j[2]]["content"]), jobs))
    ok = 0
    for (ei, si, ti), (txt, good) in zip(jobs, res):
        data[ei]["sessions"][si]["conversation"][ti]["content"] = txt
        ok += good
    json.dump(data, open(a.out, "w"), ensure_ascii=False)
    print(f"rewrote {ok}/{len(jobs)} dependency turns; kept original for {len(jobs) - ok}")


if __name__ == "__main__":
    main()
