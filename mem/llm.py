"""One call to gpt-6-luna through the local ChatGPT proxy, with a disk cache so any run can be resumed,
and a quota check so V3 never pushes the Pro account past the reserve the user asked to keep."""
import hashlib, json, os, re, threading, time, urllib.request

PORT = os.environ.get("V3_PROXY_PORT", "18319")   # 18319 = Plus-only proxy; 18318 = Pro-first pool
BASE = f"http://127.0.0.1:{PORT}/v1/responses"
MODEL = os.environ.get("V3_MODEL", "gpt-6-luna")   # V3_MODEL=deepseek-v4.1-flash runs everything on the H20
NO_GPT = os.environ.get("V3_NO_GPT") == "1"        # user, 10-14 15:40: no GPT calls until further notice
DS_URL = "http://127.0.0.1:18003/v1/chat/completions"   # H20 DeepSeek through the existing SSH tunnel
QWEN_URL = os.environ.get("V3_QWEN_URL", "http://127.0.0.1:18011/v1/chat/completions")   # Qwen judge on H20 GPUs 0-3 (rtm-judge)
READER9_URL = os.environ.get("V3_READER9_URL", "http://127.0.0.1:18012/v1/chat/completions")   # Qwen3.5-9B (LME-V2 official reader), H20 GPUs 0-3 (rtm-reader)
DS_KEY = ""
try:
    DS_KEY = re.findall(r"DEEPSEEK_KEY=(\S+)", open(os.path.expanduser("~/auto-llm-driving/deepseek-key.env")).read())[0]
except Exception:
    pass
# The fingerprint (lmfpd, 2026-10-14) shows the Plus account serves gpt-5.6-luna whatever name is requested,
# while Pro serves the real gpt-6-luna. ARM keeps the two apart in the cache and in every output file.
def on_h20(m):
    """Open models served on the H20 tunnel (chat completions): DeepSeek and Qwen."""
    return m.startswith("deepseek") or m.startswith("Qwen")


DS_VER = os.environ.get("V3_DS_VER", "")   # weight version on n03 (e.g. 1017); separates cache keys and output dirs
# V3_PLUS_REAL=1: the Plus account logged in on 10-23 serves the real gpt-6-luna (lmfpd 99.98%), so Plus runs share the
# pro6 cache keys and output directories (Pro expired on 10-23 and will not come back).
PLUS_REAL = os.environ.get("V3_PLUS_REAL") == "1"
ARM = (("ds41" + ("_" + DS_VER if DS_VER else "")) if MODEL.startswith("deepseek") else "qwen38" if MODEL.startswith("Qwen")
       else "plus56" if PORT == "18319" and not PLUS_REAL else "pro6" if MODEL == "gpt-6-luna" else "pro_" + MODEL)
# pro6 = gpt-6-luna on Pro (all earlier runs); any other Pro model gets its own output directories
STOP_AT = int(os.environ.get("V3_PRO_STOP", "97"))   # Pro weekly percent; user 10-14 night: Pro may be used up
_cfg = open(os.path.expanduser("~/.local/share/codex-workflow/llm-proxy-private/config.yaml")).read()
_KEY = re.findall(r'^\s*-\s*"?([A-Za-z0-9_\-\.]{12,})"?\s*$', _cfg[_cfg.find("api-keys"):][:800], re.M)[0]
_lock = threading.Lock()
CACHE = os.path.join(os.path.dirname(__file__), "..", "runs", "cache.jsonl")
_cache = {}
if os.path.exists(CACHE):
    for line in open(CACHE, errors="replace"):   # a line torn by a crash must not stop every run
        try:
            r = json.loads(line); _cache[r["k"]] = r
        except Exception:
            pass


def _usage(auth_file):
    t = json.load(open(auth_file))
    t = t.get("tokens", t)
    req = urllib.request.Request("https://chatgpt.com/backend-api/wham/usage", headers={
        "Authorization": "Bearer " + t["access_token"], "ChatGPT-Account-Id": t["account_id"], "User-Agent": "codex_cli_rs"})
    return json.loads(urllib.request.urlopen(req, timeout=40).read())["rate_limit"]


def pro_used():
    return _usage(os.path.expanduser("~/.codex/auth.json"))["primary_window"]["used_percent"]


def plus_used():
    import glob
    f = glob.glob(os.path.expanduser("~/.local/share/codex-workflow/llm-proxy-private/auth-plus/*plus*.json"))[0]
    w = _usage(f)
    return w["primary_window"]["used_percent"], w["secondary_window"]["used_percent"]


def h20_window():
    """H20 big-model runs only in its idle window, 11:00-20:00 US Eastern (user rule)."""
    if os.environ.get("V3_NO_WINDOW") == "1":    # user 10-17 16:1x: DeepSeek may run at full speed, no window
        return
    import datetime, zoneinfo
    h = datetime.datetime.now(zoneinfo.ZoneInfo("America/New_York")).hour
    if not 11 <= h < 20:
        raise SystemExit("outside the H20 window (11:00-20:00 US Eastern): stopping")


def check_quota():
    if on_h20(MODEL):
        h20_window()
        return 0
    """Pro: stop at STOP_AT (the user keeps the last 10%). Plus has no credit balance, so running it dry costs
    nothing; just report its 5-hour and weekly percent."""
    for attempt in range(5):        # a network hiccup on the usage endpoint must not kill a run
        try:
            u = pro_used() if PORT == "18318" else plus_used()
            break
        except Exception:
            if attempt == 4:     # usage endpoint down or its token expired: the guard only records (user rule), go on
                print("warning: Pro usage check failed; continuing without it", flush=True)
                return None
            time.sleep(5)
    if PORT == "18318" and u is not None and u >= STOP_AT:
        raise SystemExit(f"Pro weekly usage {u}% >= {STOP_AT}%: stopping")
    return u


PRO_STOP = STOP_AT
_pro_state = {"n": 0, "used": None}


def _pro_guard():
    """Re-read the Pro weekly percent every 25 Pro calls; stop the run at PRO_STOP."""
    with _lock:
        _pro_state["n"] += 1
        if _pro_state["used"] is None or _pro_state["n"] % 25 == 0:
            try:
                _pro_state["used"] = pro_used()
            except Exception:
                pass
        if _pro_state["used"] is not None and _pro_state["used"] >= PRO_STOP:
            raise SystemExit(f"Pro weekly usage {_pro_state['used']}% >= {PRO_STOP}%: stopping")


PLUS_STOP = int(os.environ.get("V3_PLUS_STOP", "97"))   # Plus weekly percent at which a run stops
_plus_state = {"n": 0, "used": None}


def _plus_guard():
    """Every 25 Plus calls re-read usage: stop at PLUS_STOP weekly; wait while the 5-hour window is nearly full."""
    with _lock:
        _plus_state["n"] += 1
        if _plus_state["used"] is None or _plus_state["n"] % 25 == 0:
            try:
                _plus_state["used"] = plus_used()
            except Exception:
                pass
        u = _plus_state["used"]
        if u is not None and u[1] >= PLUS_STOP:
            raise SystemExit(f"Plus weekly usage {u[1]}% >= {PLUS_STOP}%: stopping")
        while u is not None and u[0] >= 97:
            print(f"Plus 5-hour window at {u[0]}%: waiting 10 min", flush=True)
            time.sleep(600)
            try:
                u = _plus_state["used"] = plus_used()
            except Exception:
                pass


def call(instructions, user, effort="medium", max_tokens=16000, tag="", model=None, port=None):
    """model/port override the defaults for one call (e.g. the strong reader gpt-6.1-sol on Pro, port 18318)."""
    model, port = model or MODEL, port or PORT
    if on_h20(model):
        return _call_ds(instructions, user, effort, max_tokens, tag, model)
    if NO_GPT:
        raise SystemExit("GPT calls are switched off (V3_NO_GPT=1)")
    arm = "plus56" if port == "18319" and not PLUS_REAL else "pro6"
    if port == "18318":
        _pro_guard()
    elif port == "18319":
        _plus_guard()
    key = [model, effort, instructions, user] + ([arm] if arm != "pro6" else [])   # pro6 keeps its old keys
    if os.environ.get("V3_GEN", "1") != "1":    # independent generation n: new cache keys for every call
        key.append("gen" + os.environ["V3_GEN"])
    k = hashlib.sha256(json.dumps(key).encode()).hexdigest()
    if k in _cache:
        return _cache[k]["text"]
    body = {"model": model, "instructions": instructions, "input": [{"role": "user", "content": user}],
            "max_output_tokens": max_tokens, "reasoning": {"effort": effort}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/responses", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + _KEY})
    for attempt in range(9):            # a request that never answered is retried; it cannot bias a result
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                raw = json.loads(r.read())
            break
        except Exception as e:
            err = e.read().decode("utf-8", "replace") if isinstance(e, urllib.error.HTTPError) else ""
            if "safety_violation" in err:   # provider output filter (e.g. "regurg plugin"): deterministic, no retry;
                with _lock:                 # treated as an empty output (no records / empty answer), logged, not cached
                    open(os.path.join(os.path.dirname(CACHE), "blocked.jsonl"), "a").write(
                        json.dumps({"k": k, "tag": tag, "model": model, "err": err[:300]}) + "\n")
                print(f"warning: output blocked by provider filter ({tag})", flush=True)
                return ""
            if attempt == 8:
                raise
            time.sleep(30 * (attempt + 1))
    text = "".join(c.get("text", "") for it in raw.get("output", [])
                   if it.get("type") == "message" for c in it.get("content", []))
    u = raw.get("usage") or {}
    rec = {"k": k, "tag": tag, "arm": arm, "model": model, "text": text, "served_model": raw.get("model"),
           "in": u.get("input_tokens"), "out": u.get("output_tokens"),
           "reasoning": (u.get("output_tokens_details") or {}).get("reasoning_tokens")}
    with _lock:
        _cache[k] = rec
        open(CACHE, "a").write(json.dumps(rec, ensure_ascii=False) + "\n")
    return text


def _call_ds(instructions, user, effort, max_tokens, tag, model):
    """DeepSeek on the H20 (chat completions). effort "low" switches thinking off, anything else leaves it on."""
    key = [model, effort, instructions, user, "h20"] + ([DS_VER] if DS_VER else [])
    if os.environ.get("V3_GEN", "1") != "1":    # independent generation n, as in call()
        key.append("gen" + os.environ["V3_GEN"])
    k = hashlib.sha256(json.dumps(key).encode()).hexdigest()
    if k in _cache:
        return _cache[k]["text"]
    h20_window()
    body = {"model": model, "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": user}],
            "max_tokens": max(max_tokens, 16000)}
    if effort == "low":
        body["chat_template_kwargs"] = {"enable_thinking": False} if model.startswith("Qwen") else {"thinking": False}
    req = urllib.request.Request((READER9_URL if model.startswith("Qwen3.5") else QWEN_URL) if model.startswith("Qwen") else DS_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + DS_KEY})
    for attempt in range(6):            # the box is shared with V2; a dropped request is retried, it saw nothing
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                raw = json.loads(r.read())
            break
        except Exception:
            if attempt == 5:
                raise
            time.sleep(30 * (attempt + 1))
    m = raw["choices"][0]["message"]
    text = m.get("content") or ""
    u = raw.get("usage") or {}
    rec = {"k": k, "tag": tag, "arm": ARM, "model": model, "text": text, "served_model": raw.get("model"),
           "in": u.get("prompt_tokens"), "out": u.get("completion_tokens"),
           "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")}
    with _lock:
        _cache[k] = rec
        open(CACHE, "a").write(json.dumps(rec, ensure_ascii=False) + "\n")
    return text
