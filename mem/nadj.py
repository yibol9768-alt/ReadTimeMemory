"""H38 (PROGRESS 2026-10-16 13:21): the lychee encoder plus CCR's "never adjudicate" paragraph (no slots); same
segmentation, same everything else. Output: runs/<split>/<arm>/nadj/mem_<ci>.json
Usage: V3_BEAM_SPLIT=1m python3 -m mem.nadj build --convs 5 [--par 5]"""
import argparse, os, shutil
from concurrent.futures import ThreadPoolExecutor
from . import llm, lychee as Y, beam as B

NADJ = ("Never decide which of two inconsistent statements is right, and never drop a statement because it conflicts "
        "with an earlier one: if the user says something that contradicts or revises what was said before, write it "
        "as its own record, as said (the disambiguation state is only for who/what names refer to, not for which "
        "claim is true).\n")
Y.ENC_SYS = Y.ENC_SYS.replace('Output one JSON object per line, nothing else:\n',
                              NADJ + 'Output one JSON object per line, nothing else:\n')
assert NADJ in Y.ENC_SYS

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("step", choices=["build"])
    ap.add_argument("--convs", type=int, default=5); ap.add_argument("--par", type=int, default=5)
    B.ARGS = ARGS = ap.parse_args()
    convs = B.load(); cis = B.picked(convs)
    out = os.path.join(B.OUT, "nadj"); os.makedirs(out, exist_ok=True)
    shutil.copy(os.path.join(B.OUT, "lychee", "segments.json"), os.path.join(out, "segments.json"))
    seg = Y.segments(convs, cis, out=out, sess=B.sess)
    llm.check_quota()
    with ThreadPoolExecutor(min(ARGS.par, len(cis))) as ex:
        list(ex.map(lambda ci: Y.encode(ci, convs, seg[ci], out=out, sess=B.sess), cis))
