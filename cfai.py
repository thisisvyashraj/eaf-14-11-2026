"""Cloudflare Workers AI helper: asks a free open-weight LLM for candidate words. Needs two secrets:
CF_ACCOUNT_ID and CF_API_TOKEN (token with the 'Workers AI' permission). Optional: CF_MODELS (comma list, best first)."""
import os, re, time, json, logging
from datetime import datetime, timezone, timedelta
import aiohttp

log = logging.getLogger("trace")
ACCOUNT = os.getenv("CF_ACCOUNT_ID", "").strip()
TOKEN = os.getenv("CF_API_TOKEN", "").strip()
DEFAULT_MODELS = ("@cf/meta/llama-3.3-70b-instruct-fp8-fast,@cf/meta/llama-4-scout-17b-16e-instruct,"
                  "@cf/openai/gpt-oss-20b,@cf/qwen/qwen2.5-coder-32b-instruct")
MODELS = [m.strip() for m in os.getenv("CF_MODELS", DEFAULT_MODELS).split(",") if m.strip()]
TIMEOUT = float(os.getenv("CF_TIMEOUT", "14"))
enabled = bool(ACCOUNT and TOKEN)
cool = {}                       # model -> unix time until which we skip it
state = {"last_ok": None, "last_err": "", "calls": 0, "ok": 0}
STOP = set("the and for are but not you all any can had her was one our out day get has him his how new now old see two way who "
           "did its let put say she too use that with have this will your from they know want been much some time very when "
           "come here just like long make many more only over such take than them well were what where which while would".split())


def blocked():
    return (not enabled) or all(cool.get(m, 0) > time.time() for m in MODELS)


def _next_utc_midnight():
    n = datetime.now(timezone.utc)
    return (n.replace(hour=0, minute=1, second=0, microsecond=0) + timedelta(days=1)).timestamp()


def build_prompt(known, tried, n, hints=None):
    inf = sorted(((w, r) for w, r in known.items() if r < 100000), key=lambda kv: kv[1])
    close = inf[:14]
    far = [w for w, r in inf if r >= 15000][-14:]
    head = ("You are the brain of a bot playing a Contexto-style word game. A secret English word was chosen. For every word "
            "we guess, the game returns its RANK: how close that word is to the secret in a word-embedding model (rank 1 = "
            "the secret itself, 50 = very close, 1000 = loosely related, 20000+ = unrelated). Ranks reflect meaning, topic "
            "and association (things found together, same category, same context), not spelling. The secret is a common "
            "everyday English word (noun, verb or adjective).\n\n")
    if close and close[0][1] <= 30000:
        body = ("Closest guesses so far (word: rank):\n" + "\n".join(f"{w}: {r}" for w, r in close) +
                ("\n\nFar-away guesses (unrelated, so the secret is NOT in these areas): " + ", ".join(far) if far else "") +
                ("\n\nA statistical word-vector solver (noisy, may be wrong) currently suggests: " + ", ".join(hints) +
                 ". Use it as a hint, keep what fits the closest words, drop what does not, and add better ideas." if hints else "") +
                "\n\nAlready tried (never repeat): " + ", ".join(sorted(tried)[-80:]) +
                f"\n\nFirst write one line 'THEME:' with the single concept that best explains the closest words. Then write one "
                f"line 'WORDS:' with {n} NEW candidate words, most likely first, mixing the 2-3 most plausible themes, "
                "lowercase, single words, comma separated. Nothing else.")
    else:
        body = ("We have no useful clue yet. Far-away guesses so far: " + (", ".join(far) or "none") +
                "\nAlready tried (never repeat): " + ", ".join(sorted(tried)[-80:]) +
                f"\n\nWrite one line 'WORDS:' with {n} NEW probe words that cover as many DIFFERENT everyday areas as possible "
                "(nature, food, body, tools, places, emotions, actions, abstract ideas, people, science, sports ...), "
                "common single lowercase words, comma separated. Nothing else.")
    return head + body


def parse_words(text, tried, n):
    if not text:
        return []
    m = list(re.finditer(r"WORDS\s*:\s*(.+)", text, re.I))
    chunk = m[-1].group(1) if m else text
    out = []
    for w in re.findall(r"[A-Za-z]{3,20}", chunk):
        w = w.lower()
        if w not in tried and w not in out and w not in STOP:
            out.append(w)
    return out[:n]


def _extract(j):
    r = j.get("result") if isinstance(j, dict) else None
    if not isinstance(r, dict):
        return None
    if isinstance(r.get("response"), str):
        return r["response"]
    ch = r.get("choices")
    if ch:
        msg = ch[0].get("message") or {}
        return msg.get("content") or ch[0].get("text")
    out = r.get("output")
    if isinstance(out, list):                               # responses-style output (gpt-oss)
        parts = []
        for o in out:
            for c in (o.get("content") or []):
                if isinstance(c, dict) and c.get("text"):
                    parts.append(c["text"])
        return "\n".join(parts) or None
    return None


async def ask(http, prompt):
    """try the models in order until one answers; returns text or None"""
    for model in MODELS:
        if cool.get(model, 0) > time.time():
            continue
        url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run/{model}"
        state["calls"] += 1
        try:
            async with http.post(url, headers={"Authorization": f"Bearer {TOKEN}"},
                                 json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 400, "temperature": 0.5},
                                 timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
                raw = await resp.text()
                try:
                    j = json.loads(raw)
                except Exception:
                    j = {}
                if resp.status == 200 and j.get("success", True):
                    txt = _extract(j)
                    if txt:
                        state["ok"] += 1; state["last_ok"] = model
                        return txt
                    state["last_err"] = f"{model}: empty answer"; cool[model] = time.time() + 600
                    continue
                err = json.dumps(j.get("errors") or raw[:160])
                state["last_err"] = f"{model}: HTTP {resp.status} {err[:120]}"
                if resp.status == 429 or "4006" in err or "daily" in err.lower() or "allocation" in err.lower() or "neurons" in err.lower():
                    until = _next_utc_midnight()
                    for m in MODELS:
                        cool[m] = until               # the free neuron pool is shared by all models
                    log.warning("Cloudflare AI: free daily quota used up - LLM off until UTC midnight (%s)", err[:100])
                    return None
                if resp.status in (401, 403):
                    for m in MODELS:
                        cool[m] = time.time() + 3600
                    log.error("Cloudflare AI: token/account rejected (HTTP %s) - check CF_ACCOUNT_ID / CF_API_TOKEN (needs Workers AI permission)", resp.status)
                    return None
                cool[model] = time.time() + (3600 if resp.status in (400, 404) else 90)
                log.warning("Cloudflare AI model %s failed: %s", model, state["last_err"])
        except Exception as e:
            state["last_err"] = f"{model}: {e!r}"[:160]
            cool[model] = time.time() + 60
            log.warning("Cloudflare AI model %s error: %r", model, e)
    return None


async def propose(http, known, tried, n=12, hints=None):
    txt = await ask(http, build_prompt(known, tried, n, hints))
    words = parse_words(txt, tried, n)
    if words:
        log.info("LLM proposes: %s", ", ".join(words))
    return words
