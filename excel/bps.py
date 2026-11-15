"""Minimal client for the ChatGPT-for-Excel backend (bps.openai.com), using the Plus account's Codex login.
The user chose this route for the Plus account (2026-10-14), knowing it is unofficial and may break the terms.
The token is sent only to bps.openai.com."""
import glob, json, os, sys, urllib.request

URL = "https://bps.openai.com/basispoints/api/responses"
AUTH = glob.glob(os.path.expanduser("~/.local/share/codex-workflow/llm-proxy-private/auth-plus/*plus*.json"))[0]


def call(model, instructions, user, effort="medium", timeout=600):
    a = json.load(open(AUTH))
    headers = {"accept": "application/json", "accept-encoding": "identity", "content-type": "application/json",
               "origin": "https://bps.openai.com", "authorization": "Bearer " + a["access_token"],
               "chatgpt-account-id": a["account_id"], "x-openai-account-id": a["account_id"],
               "x-basispoints-auth-mode": "chatgpt",
               "x-openai-internal-basispoints-client-product": "basispoints-excel-plugin",
               "x-stainless-package-version": "6.31.0", "x-stainless-runtime": "browser:chrome"}
    body = {"model": model, "model_selection": "explicit", "stream": False, "store": False,
            "reasoning_effort": effort,
            "input": [{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": instructions}]},
                      {"type": "message", "role": "user", "content": [{"type": "input_text", "text": user}]}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=headers)
    try:
        raw = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:600]!r}")
    text = "".join(c.get("text", "") for it in raw.get("output", []) if it.get("type") == "message"
                   for c in it.get("content", []))
    return text, raw


if __name__ == "__main__":
    t, raw = call(sys.argv[1] if len(sys.argv) > 1 else "gpt-6-luna", "Answer briefly.", "What is 23*17?")
    print(repr(t)); print(raw.get("model"), json.dumps(raw.get("usage"))[:400])
    print("top-level keys:", list(raw.keys()))
