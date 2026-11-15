"""Replay memory for web-agent trajectories: a non-parametric world model built by a program (no LLM, no training).

Write side
  keyframes: every canonical page (title + normalised URL path) gets ONE reconstructed accessibility tree, the
             order-preserving union of every visit (new nodes are aligned into place with a sequence matcher); each
             node remembers in how many of the page's visits it was seen.
  deltas:    every step s_t --a_t--> s_t+1 stores which named nodes appeared / disappeared, with the task goal,
             the agent's thought, the action and the pages before/after.
Read side (replay)
  The question is expanded into literal UI terms (same reader model). Pages are scored by IDF-weighted term hits on
  their keyframe; each chosen page is replayed in the reader's native accessibility-tree format, pruned to the matching
  nodes plus their ancestors and neighbouring siblings (order kept). Matching deltas are replayed as
  "on PAGE, did ACTION -> appeared / disappeared". A page's visit count is shown as coverage.

Usage: python3 -m mem.replay build --traj FILE --ids-from HAYSTACK.json --domain web --out DIR
"""
import argparse, collections, difflib, json, math, os, re, pickle
from .uistore import LINE, norm_url, expanded_keywords

NOISE = {"StaticText", "LineBreak", "generic", "none", "InlineTextBox"}


def nodes_of(tree):
    """-> list of (depth, role, name, props) in document order; StaticText kept only if its parent has no name."""
    out = []
    for raw in (tree or "").split("\n"):
        m = LINE.match(raw)
        if not m:
            continue
        depth, _, role, n1, n2, props = m.groups()
        name = (n1 if n1 is not None else n2 or "").replace("\\'", "'").strip()
        if role in NOISE and role != "StaticText":
            continue
        if role == "StaticText" and (not name or (out and out[-1][2] and name in out[-1][2])):
            continue
        keep = re.findall(r"(checked='?\w+'?|selected=\w+|expanded=\w+|disabled=\w+|value='[^']{0,60}')", props or "")
        out.append((len(depth), role, name[:200], " ".join(keep)))
    return out


def signatures(nodes):
    """Stable node identity: role + name + named-ancestor path (ids/bids differ across visits)."""
    sig, stack = [], []
    for d, r, n, _ in nodes:
        while stack and stack[-1][0] >= d:
            stack.pop()
        path = "/".join(f"{rr}:{nn[:40]}" for _, rr, nn in stack if nn)
        sig.append(f"{path}|{r}:{n}")
        stack.append((d, r, n))
    return sig


def page_key(title, url):
    """Page class: title with record numbers masked (INC0010006 -> <REC>) + normalised URL path."""
    t = re.sub(r"#\d{5,}\b", "#<REC>", re.sub(r"\b[A-Z]{1,6}\d{5,}\b", "<REC>", title))
    return f"{t} [{norm_url(url)}]"


ROLES = {"tab": {"tab"}, "link": {"link"}, "button": {"button"}, "checkbox": {"checkbox", "menuitemcheckbox"},
         "dropdown": {"combobox", "listbox"}, "radio": {"radio", "menuitemradio"},
         "field": {"textbox", "combobox", "checkbox", "spinbutton", "searchbox", "radio"},
         "column": {"columnheader"}, "row": {"row"}, "menuitem": {"menuitem", "menuitemcheckbox", "menuitemradio"},
         "option": {"option"}, "heading": {"heading"}, "image": {"img", "image"}, "text": {"StaticText", "paragraph"}}
CLAIM_SYS = ("You prepare checks against a memory of web-app screens (accessibility trees). Rewrite what the question "
             "assumes or asks about into 1-4 census queries that a program can answer by counting/listing UI elements "
             "on recorded pages. Each query: {\"page\": words naming the page/form/list/dialog as its title would, "
             "\"role\": one of " + str(sorted(ROLES) + ["any"]) + ", \"name_contains\": a literal substring of the "
             "element label or \"\"}. Prefer queries whose answer would confirm or refute the question's premise "
             "(e.g. how many tabs a form has, which links exist after an action, whether any checkbox label contains X). "
             "Output JSON {\"queries\": [...]} only.")


class Keyframe:
    """Order-preserving union of all visits of one page. A node unseen so far is inserted right after the last known
    node that preceded it in the current visit (fractional order keys, linear time)."""
    def __init__(self):
        self.key, self.node, self.cnt, self.visits = {}, {}, collections.Counter(), 0

    def merge(self, nodes):
        self.visits += 1
        sig = signatures(nodes)
        prev_key, seen_now = 0.0, set()
        for i, sg in enumerate(sig):
            if sg in seen_now:
                continue
            seen_now.add(sg)
            if sg in self.key:
                prev_key = self.key[sg]
            else:
                nxt = [k for k in self._succ(prev_key)]
                k = (prev_key + nxt[0]) / 2 if nxt else prev_key + 1.0
                self.key[sg], self.node[sg] = k, nodes[i]
                prev_key = k
            self.cnt[sg] += 1

    def _succ(self, k):
        # smallest key greater than k (cached sorted keys rebuilt lazily)
        if getattr(self, "_sorted_n", -1) != len(self.key):
            self._sorted = sorted(self.key.values()); self._sorted_n = len(self.key)
        import bisect
        j = bisect.bisect_right(self._sorted, k)
        out = [self._sorted[j]] if j < len(self._sorted) else []
        return out

    def finalize(self):
        order = sorted(self.key, key=self.key.get)
        self.sigs = order
        self.nodes = [self.node[s] for s in order]
        self.seen = [self.cnt[s] for s in order]
        del self.key, self.node, self.cnt
        for a in ("_sorted", "_sorted_n"):
            if hasattr(self, a):
                delattr(self, a)


class ReplayStore:
    def __init__(self):
        self.pages = {}      # page -> Keyframe
        self.deltas = []     # dict: traj, goal, outcome, i, page, next_page, action, thought, appeared, disappeared

    def add_traj(self, t):
        prev = None
        for s in t["states"]:
            nodes = nodes_of(s.get("accessibility_tree"))
            title = next((n for _, r, n, _ in nodes if r == "RootWebArea"), "")
            page = page_key(title, s.get("url"))
            kf = self.pages.setdefault(page, Keyframe())
            kf.merge(nodes)
            cur = {f"{r} '{n}'" for _, r, n, _ in nodes if n and r != "StaticText"}
            if prev is not None:
                pp, pnodes, pthought, pact = prev
                d = {"traj": t["id"], "goal": t.get("goal", ""), "outcome": t.get("outcome"), "i": s["state_index"],
                     "page": pp, "next_page": page, "action": s.get("action"), "thought": (s.get("thought") or "")[:500],
                     "appeared": sorted(cur - pnodes)[:40], "disappeared": sorted(pnodes - cur)[:20]}
                self.deltas.append(d)
            prev = (page, cur, s.get("thought"), s.get("action"))

    def save(self, d):
        for kf in self.pages.values():
            kf.finalize()
        os.makedirs(d, exist_ok=True)
        pickle.dump({"pages": {p: (kf.nodes, kf.seen, kf.visits) for p, kf in self.pages.items()},
                     "deltas": self.deltas}, open(os.path.join(d, "replay.pkl"), "wb"))


def render(nodes, seen, visits, keep):
    lines = []
    for idx in sorted(keep):
        d, r, n, p = nodes[idx]
        tag = f" (seen {seen[idx]}/{visits})" if visits > 1 and seen[idx] < visits else ""
        lines.append("\t" * min(d, 12) + (f"{r} '{n}'" if n else r) + (f" {p}" if p else "") + tag)
    return "\n".join(lines)


class ReplayIndex:
    def __init__(self, d):
        st = pickle.load(open(os.path.join(d, "replay.pkl"), "rb"))
        self.deltas = st["deltas"]
        self.pages = {}
        for p, (nodes, seen, visits) in st["pages"].items():
            kf = Keyframe(); kf.nodes, kf.seen, kf.visits = nodes, seen, visits
            self.pages[p] = kf
        self.ptext = {p: (p + "\n" + "\n".join(f"{r} {n}" for _, r, n, _ in kf.nodes)).lower() for p, kf in self.pages.items()}

    def absent(self, kws):
        """Terms that occur nowhere in the reconstructed pages (union of every visit) nor in any replayed step."""
        if not hasattr(self, "_dtext"):
            self._dtext = "\n".join((d["goal"] + " " + str(d["action"]) + " " + d["thought"]).lower() for d in self.deltas)
            self._all = "\n".join(self.ptext.values())
        return [k for k in kws if len(k) > 2 and k.lower() not in self._all and k.lower() not in self._dtext]

    def presupposition(self, q, max_terms=8):
        """Check only the names the question itself states (quoted strings, capitalised UI names), not expansions.
        Each is reported as found (where, how often) or not found anywhere."""
        from .uistore import keywords
        if not hasattr(self, "_dtext"):
            self.absent([])
        qq = q.split("Mark your final answer")[0]
        quoted = re.findall(r"[`\"“']([^`\"”']{2,60})[`\"”']", qq)
        caps = [k for k in keywords(qq) if k[:1].isupper() and k not in quoted]
        generic = {"servicenow", "i", "our", "magento", "reddit", "postmill", "company"}
        terms, seen = [], set()
        for t in quoted + caps:
            tl = t.lower().strip()
            if len(tl) < 3 or tl in generic or tl in seen:
                continue
            seen.add(tl); terms.append(t.strip())
        terms = terms[:max_terms]
        if not terms:
            return ""
        lines = []
        for t in terms:
            tl = t.lower()
            where = [(kf.visits, p) for p, kf in self.pages.items() if tl in self.ptext[p]]
            if where:
                where.sort(reverse=True)
                lines.append(f"- '{t}': found on {len(where)} page(s), e.g. " +
                             "; ".join(f"{p[:70]} ({v} visits)" for v, p in where[:2]))
            elif tl in self._dtext:
                lines.append(f"- '{t}': not on any page, but mentioned in past steps (goals/thoughts/actions)")
            else:
                lines.append(f"- '{t}': NOT FOUND in any reconstructed page or past step")
        return ("Name check (each name stated in the question vs. everything ever observed in "
                f"{sum(kf.visits for kf in self.pages.values())} page visits):\n" + "\n".join(lines))

    def census_queries(self, q):
        m = os.environ.get("V3_LME_EXPAND_MODEL")
        if not m:
            return []
        from . import llm
        import json as _j
        txt = llm.call(CLAIM_SYS, q.split("Mark your final answer")[0], effort="low", max_tokens=1200,
                       tag="lme_census", model=m)
        mm = re.search(r"\{.*\}", txt or "", re.S)
        try:
            qs = _j.loads(mm.group(0)).get("queries", [])
        except Exception:
            return []
        return [x for x in qs if isinstance(x, dict) and x.get("page")][:4]

    def census(self, cq, max_pages=3):
        """Exact counts/lists of elements of one role on the best-matching recorded pages (union of all visits)."""
        words = [w for w in re.findall(r"[a-z0-9]+", str(cq.get("page", "")).lower()) if len(w) > 2]
        roles = ROLES.get(str(cq.get("role", "any")).lower())
        sub = str(cq.get("name_contains") or "").lower().strip()
        cand = []
        want_form = "form" in words or "record" in words
        for p, kf in self.pages.items():
            title = p.lower()
            tw = set(re.findall(r"[a-z0-9]+", title))
            sc = sum(1 for w in words if w in tw or (w.endswith("s") and w[:-1] in tw) or w + "s" in tw)
            if not sc:
                continue
            if want_form and "_list.do" in title:
                sc -= 0.5
            idx = [i for i, (_, r, n, _) in enumerate(kf.nodes) if n and (roles is None or r in roles)]
            has_sub = any(sub in kf.nodes[i][2].lower() for i in idx) if sub else False
            # pages that actually have elements of the asked role (and the asked label) come first
            cand.append((sc, has_sub, bool(idx), kf.visits, p, idx))
        cand.sort(key=lambda x: x[:4], reverse=True)
        out = []
        for sc, _, _, v, p, idx in cand[:max_pages]:
            kf = self.pages[p]
            role_txt = cq.get("role", "any")
            if sub:
                hit = [i for i in idx if sub in kf.nodes[i][2].lower()]
                if hit:
                    out.append(f"- {p[:90]} ({v} visits): {len(hit)} {role_txt} element(s) contain '{sub}': " +
                               ", ".join(f"'{kf.nodes[i][2][:60]}' (seen {kf.seen[i]}/{v})" for i in hit[:12]))
                else:
                    out.append(f"- {p[:90]} ({v} visits): none of its {len(idx)} {role_txt} elements contains '{sub}'")
            else:
                names = []
                for i in idx:
                    nm = kf.nodes[i][2][:50]
                    if nm not in names:
                        names.append(nm)
                lo = min((kf.seen[i] for i in idx), default=0)
                out.append(f"- {p[:90]} ({v} visits): {len(names)} distinct {role_txt} element(s)" +
                           (f" (each seen in >= {lo}/{v} visits)" if names else "") +
                           (": " + "; ".join(names[:30]) if names else ""))
        if not cand:
            out.append(f"- no recorded page title matches '{cq.get('page')}'")
        return out

    def fact_sheet(self, q):
        cqs = self.census_queries(q)
        if not cqs:
            return ""
        lines = []
        for cq in cqs:
            lines.append(f"Query: {cq.get('role', 'any')} elements" +
                         (f" whose label contains '{cq.get('name_contains')}'" if cq.get("name_contains") else "") +
                         f" on '{cq.get('page')}'")
            lines += self.census(cq)
        return ("Fact sheet (computed by the memory program over every recorded visit of each page; counts and lists "
                "are exact for what was observed):\n" + "\n".join(lines))

    def query(self, q, budget_chars=60000, sib=3, coverage=False):
        kws = expanded_keywords(q)
        lk = [k.lower() for k in kws if len(k) > 1]
        N = len(self.pages)
        df = {k: sum(1 for t in self.ptext.values() if k in t) for k in lk}
        w = {k: math.log(1 + N / (1 + df[k])) * (1.5 if " " in k else 1.0) for k in lk}
        scored = sorted(((sum(w[k] for k in lk if k in t), p) for p, t in self.ptext.items()), reverse=True)
        out, used = [], 0
        if coverage == "facts":
            fs = self.fact_sheet(q)
            if fs:
                fs = fs[:6000]; out.append(fs); used += len(fs)
        elif coverage == "names":
            pc = self.presupposition(q)
            if pc:
                out.append(pc); used += len(pc)
        elif coverage:
            nv = sum(kf.visits for kf in self.pages.values())
            cov = (f"Memory = {len(self.pages)} pages reconstructed from all {nv} page visits in past trajectories "
                   f"(every element ever seen on a page is in its reconstruction).")
            ab = self.absent(kws)
            if ab:
                cov += " Exhaustive check: these terms appear NOWHERE in any reconstructed page or past step: " + \
                       ", ".join(repr(a) for a in ab[:12]) + "."
            out.append(cov); used += len(cov)
        page_budget = budget_chars * 0.65
        for sc, p in scored:
            if sc <= 0 or used > page_budget:
                break
            kf = self.pages[p]
            hits = []
            for i, (_, r, n, _) in enumerate(kf.nodes):
                t = (r + " " + n).lower()
                hw = sum(w[k] for k in lk if k in t)
                if hw > 0:
                    hits.append((hw, i))
            if not hits:
                continue
            hits.sort(key=lambda x: -x[0])
            cap = min(page_budget / 4, page_budget - used)
            keep, depth = set(), [nd[0] for nd in kf.nodes]
            for hw, i in hits:
                add = set(range(max(0, i - sib), min(len(kf.nodes), i + sib + 1)))
                anc, dd, j = [], depth[i], i - 1
                while j >= 0 and dd > 0:                 # ancestors, nearest first
                    if depth[j] < dd:
                        add.add(j); dd = depth[j]; anc.append(j)
                    j -= 1
                for a in anc:                            # the largest enclosing block of <= 150 nodes (a row, a form)
                    e = a + 1
                    while e < len(kf.nodes) and depth[e] > depth[a] and e - a <= 150:
                        e += 1
                    if e - a > 150:
                        break
                    add.update(range(a, e))
                if kf.nodes[i][1] in ("dialog", "alertdialog", "radiogroup", "listbox", "menu", "table", "grid",
                                      "tablist", "form", "combobox", "list", "row", "group"):
                    j = i + 1
                    while j < len(kf.nodes) and depth[j] > depth[i] and j - i < 120:
                        add.add(j); j += 1
                new = keep | add
                if len(render(kf.nodes, kf.seen, kf.visits, new)) > cap and keep:
                    continue
                keep = new
            block = f"\n## Reconstructed page: {p} (visited {kf.visits} times)\n" + render(kf.nodes, kf.seen, kf.visits, keep)
            block = block[: int(cap) + 200]
            out.append(block); used += len(block)
        ds = []
        for d in self.deltas:
            text = (d["goal"] + " " + str(d["action"]) + " " + d["thought"] + " " + " ".join(d["appeared"])).lower()
            sc = sum(w[k] for k in lk if k in text)
            if sc > 0:
                ds.append((sc, d))
        ds.sort(key=lambda x: -x[0])
        out.append("\n## Replayed steps (task | outcome) on PAGE: ACTION -> appeared / disappeared; agent's thought")
        for sc, d in ds:
            line = (f"\n- [{d['goal'][:140]} | {d['outcome']}] step {d['i']} on {d['page'][:70]}: {d['action']}"
                    f" -> {d['next_page'][:60]}; appeared: {', '.join(d['appeared'][:12])[:400]}"
                    + (f"; disappeared: {', '.join(d['disappeared'][:6])[:200]}" if d["disappeared"] else "")
                    + f" | thought: {d['thought'][:250]}")
            if used + len(line) > budget_chars:
                break
            out.append(line); used += len(line)
        return "\n".join(out), kws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build"])
    ap.add_argument("--traj"); ap.add_argument("--ids-from"); ap.add_argument("--domain"); ap.add_argument("--out")
    a = ap.parse_args()
    hs = json.load(open(a.ids_from))
    qdom = {json.loads(l)["id"]: json.loads(l)["domain"]
            for l in open(os.path.join(os.path.dirname(os.path.dirname(a.ids_from)), "questions.jsonl"))}
    need = set().union(*[set(v) for q, v in hs.items() if qdom[q] == a.domain])
    st = ReplayStore()
    for l in open(a.traj):
        t = json.loads(l)
        if t["id"] in need:
            st.add_traj(t)
    st.save(a.out)
    print(f"{a.domain}: {len(st.pages)} pages, {sum(len(k.nodes) for k in st.pages.values())} keyframe nodes, "
          f"{len(st.deltas)} deltas")


if __name__ == "__main__":
    main()
