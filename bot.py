"""
Trace auto-player v3 (Telethon userbot) for Iris' ".trace" word game.

What changed in v3
  * ONE guess at a time: after every reply from Iris the solver re-plans, instead of sending 5 stale words in a row.
  * Each word we send is remembered by its Telegram message id, so Iris' "already taken" / "Not a word I recognize"
    replies are matched to the exact word and that word is never sent again in the round (no more repeated words).
  * Round tracking only reacts to real Trace messages (a Traceboard / music / aura message can no longer be mistaken
    for "round over"), so it no longer sends .trace while a round is running.
  * Answers of finished rounds are saved to answers.json and fed back to the solver (a repeated word is found faster).
  * Round older than 5 min with no word better than #5000 -> ".trace end" and a fresh round.
  * 3 min rest after every 8th finished round. No automatic quiet hours: the bot only stops when YOU pause it.
  * Pause / resume / status from: Telegram (Saved Messages of the bot account, or OWNER_ID), a web link
    (CONTROL_KEY) and the console window.
All secrets come from environment variables.
"""
import asyncio, os, re, random, time, logging, json, sys, threading, collections
from datetime import datetime, timedelta, timezone
import aiohttp
from aiohttp import web
from telethon import TelegramClient, events, errors
from telethon.sessions import StringSession
import cfai                                         # Cloudflare Workers AI (free LLM) helper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("trace")


class _Ring(logging.Handler):
    """keeps the last 400 log lines in memory so they can be read at /debug?key=CONTROL_KEY"""
    def __init__(self):
        super().__init__(); self.buf = collections.deque(maxlen=400)

    def emit(self, record):
        try:
            self.buf.append(self.format(record))
        except Exception:
            pass


RING = _Ring(); RING.setFormatter(logging.Formatter("%(asctime)s %(message)s")); logging.getLogger().addHandler(RING)

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]
SESSION = os.environ["SESSION_STRING"]
GROUP = os.environ["GROUP"].strip()               # group id (-100...) or @username where Iris lives
if GROUP.lstrip("-").isdigit():
    GROUP = int(GROUP)                            # numeric ids must be int, not str
GROQ_KEY = os.getenv("GROQ_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")

SEND_DELAY = float(os.getenv("SEND_DELAY", "2.8"))     # min seconds between two guesses (+0-0.3 s jitter). Slows itself down after a FloodWait
REPLY_WAIT = float(os.getenv("REPLY_WAIT", "5"))       # how long to wait for Iris' answer to one guess
PAUSE_EVERY = int(os.getenv("PAUSE_EVERY", "8"))        # finished rounds between breaks
PAUSE_SECONDS = int(os.getenv("PAUSE_SECONDS", "180"))  # break length (3 min)
STALE_SECONDS = int(os.getenv("STALE_SECONDS", "300"))  # round older than this ...
STALE_RANK = int(os.getenv("STALE_RANK", "5000"))       # ... and still no word ranked better than this -> .trace end
ANSWERS_FILE = os.getenv("ANSWERS_FILE", "answers.json")
CONTROL_KEY = os.getenv("CONTROL_KEY", "").strip()      # enables the web controls: /pause?key=...  /resume?key=...  /status?key=...
def _ids(*names):
    out = set()
    for n in names:
        for x in re.split(r"[,\s;]+", os.getenv(n, "") or ""):
            if x.strip().lstrip("-").isdigit() and int(x) != 0:
                out.add(int(x))
    return out
OWNERS = _ids("OWNER_IDS", "OWNER_ID")                  # numeric Telegram ids of everyone allowed to control the bot (comma separated)
STOP_AFTER_ROUNDS = int(os.getenv("STOP_AFTER_ROUNDS", "0") or 0)      # optional timer at startup: pause after N finished rounds
STOP_AFTER_HOURS = float(os.getenv("STOP_AFTER_HOURS", "0") or 0)      # ... and/or after N hours
VERSION = "4.1"
LEGACY_LLM = os.getenv("LEGACY_LLM", "0") == "1"           # old Groq/Gemini/OpenRouter helpers: off by default (quota-exhausted, only spam the log)
TRACE_BACKOFF_BASE = float(os.getenv("TRACE_BACKOFF_BASE", "60"))   # after an unanswered .trace wait 60 s, then 120, 240 ... (never spam)
TRACE_BACKOFF_MAX = float(os.getenv("TRACE_BACKOFF_MAX", "600"))
STALE_MIN_GUESSES = int(os.getenv("STALE_MIN_GUESSES", "15"))      # the 5-minute abandon rule only applies after this many guesses
LLM_EVERY = int(os.getenv("CF_EVERY", "3"))            # ask the LLM for fresh candidates every N guesses
LLM_MARGIN = float(os.getenv("CF_MARGIN", "1.0"))      # LLM word is used unless the solver's own pick scores this much better
TRACE_GAP = float(os.getenv("TRACE_GAP", "20"))          # min seconds between two ".trace" messages
ROUND_SECONDS = 45 * 60

SEEDS = """time person water money animal food music house work nature city car book school family
game love war sport color light sound body health tree sea sky fire earth science art door street
machine king computer doctor ocean mountain bird fish flower church energy law power paper
train market garden island weather story language office window kitchen hospital soldier teacher
farmer engine medicine clothing metal stone glass river road bridge ship plane phone camera
dance movie ghost magic dream night morning winter summer rain wind snow gold silver iron""".split()

client = TelegramClient(StringSession(SESSION), API_ID, API_HASH)
ME = {"name": "", "id": None, "username": ""}


class S:                       # all bot state
    active = False             # a Trace round is running (as far as we know)
    started = 0.0              # when this round started (or when we joined it)
    last_iris = time.time()
    known = {}                 # word -> best rank seen (own guesses, other players, top-10 lists, hints)
    tried = set()              # words not to send again this round
    ids = {}                   # telegram message id of our guess -> word
    pending = None             # the word we just sent and are waiting for Iris' answer on
    round_guesses = 0
    games = 0                  # finished rounds (complete / expired / ended)
    my_wins = 0
    paused = False
    paused_at = 0.0
    rested_at_games = 0
    last_trace = 0.0
    trace_fail = 0
    extra = 0.0                # adaptive extra delay: +2s per FloodWait, -0.5s per 40 clean sends
    sent = 0
    dialogs_loaded = False
    abandon_blocked = False    # ".trace end" did not work (not the host?) -> just keep playing this round
    just_ended = 0.0
    finished_at = 0.0
    counted = set()            # message ids of TRACE COMPLETE messages already counted
    answers = []               # answers of past rounds (newest last)
    reply_evt = None           # set when Iris answered our pending guess
    ack_evt = None             # set on any Trace message from Iris
    round_id = 0
    trace_streak = 0           # .trace sent in a row without ANY answer from Iris (drives the back-off)
    trace_sends = 0
    last_sent = 0.0
    last_memlog = 0.0
    last_gameplay = 0.0        # last time a guess/top-10 message was seen (to tell a duplicate 'session started' from a real new round)
    llm_best = 10 ** 9         # best rank at the time of the last LLM request
    session_msg_id = None      # id of the 'Group session started' message of the current round (edits of it must NOT reset the round)
    timer_rounds = None        # pause when this many rounds have finished in total
    timer_until = None         # pause at this unix time
    why_inactive = "program start"   # last reason the round was considered over (shown when .trace is sent)
    boot = time.time()
    llm_words = []             # candidates from the LLM for the current round
    llm_task = None
    llm_last = -99


def reset_round():
    S.known, S.tried, S.ids = {}, set(), {}
    S.pending = None
    S.round_guesses = 0
    S.started = time.time()
    S.active = True
    S.abandon_blocked = False
    S.round_id += 1; S.llm_words = []; S.llm_last = -99
    m = EMB.get("model")
    if m is not None:
        m.new_round()


def _rss_now():
    """current memory of this process in MB (Linux /proc), -1 if unknown"""
    try:
        with open("/proc/self/status") as fh:
            for ln in fh:
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1]) / 1024
    except Exception:
        pass
    return -1


def _events():
    if S.reply_evt is None:
        S.reply_evt, S.ack_evt = asyncio.Event(), asyncio.Event()


# ---------------- answer memory ----------------
def load_answers():
    try:
        with open(ANSWERS_FILE, encoding="utf8") as fh:
            S.answers = [str(w).lower() for w in json.load(fh) if str(w).isalpha()]
    except FileNotFoundError:
        S.answers = []
    except Exception as e:
        log.warning("could not read %s: %r", ANSWERS_FILE, e); S.answers = []
    log.info("answer memory: %d known answers", len(S.answers))


def remember_answer(word):
    word = (word or "").lower()
    if not word.isalpha():
        return
    if word not in S.answers:
        S.answers.append(word)
        try:
            with open(ANSWERS_FILE, "w", encoding="utf8") as fh:
                json.dump(S.answers, fh)
        except Exception as e:
            log.warning("could not save %s: %r", ANSWERS_FILE, e)
    m = EMB.get("model")
    if m is not None:
        m.set_history(S.answers)


RANK_LINE = re.compile(r"#(\d+)\s*[•·]\s*([A-Za-z]+)")
CURRENT = re.compile(r"Current guess:\s*([A-Za-z]+)\s*[•·]\s*(.*?)\s*[•·]\s*\S+\s*#(\d+)", re.S)
CURRENT2 = re.compile(r"Current guess:\s*([A-Za-z]+)\s*[•·].*?#(\d+)", re.S)
COMPLETE_RE = re.compile(r"TRACE COMPLETE", re.I)
EXPIRED_RE = re.compile(r"session expired|expired after", re.I)
ENDED_RE = re.compile(r"(session|trace|game|round)\s+(has\s+been\s+|was\s+|has\s+)?(ended|cancel+ed|stopped|terminated)|ended by", re.I)
TAKEN_RE = re.compile(r"already taken", re.I)
NOTWORD_RE = re.compile(r"not a word|don'?t recogni[sz]e|not recogni[sz]ed|invalid word", re.I)
WINNER_RE = re.compile(r"Winner:\s*(.+)")
ANSWER_RE = re.compile(r"Answer:\s*([A-Za-z]+)")
PLAIN_WORD = re.compile(r"^[A-Za-z]{2,25}$")
MARKERS = ("TRACE", "already taken", "not a word", "recogni", "invalid word")

# ---------------- LLM layer (free light models, rotating) ----------------
cooldown = {}
models = {}          # provider -> model ids, best first (auto-discovered)
models_at = {}
BAD = re.compile(r"whisper|tts|guard|embed|image|audio|live|vision|imagen|veo|robot|computer|moderation|safeguard|orpheus|playai|transcri|rerank", re.I)


def _score(order, mid):
    m = mid.lower()
    for i, k in enumerate(order):
        if k in m:
            return i
    return len(order)



PROMPT = """We are playing a word-closeness game (like Contexto). A secret English word exists.
Every guess gets a rank: lower rank = semantically closer to the secret (rank 1 = the secret word).
Closest known guesses (rank: word):
{best}
Guesses that were unrelated/far: {far}
Already tried (never repeat): {tried}
Suggest {n} NEW single lowercase English words (common dictionary words, no plurals of tried words).
Method: first work out what single concept, theme or category the best-ranked words (especially the top 3)
have in common, and what the secret word could be if it sits near ALL of them. Then propose words that fit
that idea: the likely secret itself, its synonyms, parents/categories, parts, typical contexts and close
neighbours. Rank under 100 means very close: stay tightly inside that neighbourhood and try specific
candidate answers. Rank 100-1500: stay around the theme. Above 3000: explore different themes instead.
Every word must be a real, common English dictionary word in its base form (like 'bark', 'tree'); never invent words, never add endings such as -est, -men, -man, -ad to make new words.
Reply with ONLY a JSON array of strings."""


async def discover(http, p):
    """Ask the provider which models exist right now and pick small/free ones (best first)."""
    ids = []
    try:
        to = aiohttp.ClientTimeout(total=20)
        if p == "groq":
            async with http.get("https://api.groq.com/openai/v1/models",
                                headers={"Authorization": f"Bearer {GROQ_KEY}"}, timeout=to) as r:
                d = await r.json(content_type=None)
            ids = [m["id"] for m in d.get("data", []) if not BAD.search(m["id"])]
            ids.sort(key=lambda x: _score(["8b-instant", "gpt-oss-20b", "gemma", "llama-3.3", "qwen", "llama", "gpt-oss"], x))
        elif p == "gemini":
            async with http.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                                headers={"x-goog-api-key": GEMINI_KEY}, timeout=to) as r:
                d = await r.json(content_type=None)
            for m in d.get("models", []):
                mid = m.get("name", "").split("/")[-1]
                if "generateContent" in m.get("supportedGenerationMethods", []) and mid and not BAD.search(mid):
                    ids.append(mid)
            def g(x):
                s = 0 if "flash-lite" in x else 1 if "flash" in x else 2
                return (s + (0.5 if re.search(r"preview|exp|thinking", x) else 0), x)
            ids.sort(key=g)
        elif p == "openrouter":
            async with http.get("https://openrouter.ai/api/v1/models", timeout=to) as r:
                d = await r.json(content_type=None)
            for m in d.get("data", []):
                pr = m.get("pricing") or {}
                try:
                    free = float(pr.get("prompt", 1)) == 0 and float(pr.get("completion", 1)) == 0
                except (TypeError, ValueError):
                    free = False
                outs = (m.get("architecture") or {}).get("output_modalities") or ["text"]
                if free and "text" in outs and not BAD.search(m["id"]):
                    ids.append(m["id"])                       # ONLY models priced at 0
            ids.sort(key=lambda x: _score(["gpt-oss-20b", "gemma", "llama-3.2", "llama-3.3", "mistral", "qwen", "llama"], x))
    except Exception as e:
        log.warning("%s model discovery failed: %r", p, e)
    override = os.getenv({"groq": "GROQ_MODEL", "gemini": "GEMINI_MODEL", "openrouter": "OR_MODEL"}[p])
    if override:
        ids.insert(0, override)
    models[p], models_at[p] = ids, time.time()
    log.info("%s models to try: %s", p, ids[:4])


async def call(http, p, model, prompt):
    """-> (status, text, retry_after_seconds)"""
    if p == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        hdr = {"x-goog-api-key": GEMINI_KEY}
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.8, "maxOutputTokens": 600}}
    else:
        url = ("https://api.groq.com/openai/v1/chat/completions" if p == "groq"
               else "https://openrouter.ai/api/v1/chat/completions")
        hdr = {"Authorization": f"Bearer {GROQ_KEY if p == 'groq' else OPENROUTER_KEY}"}
        body = {"model": model, "temperature": 0.8, "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}]}
    async with http.post(url, headers=hdr, json=body, timeout=aiohttp.ClientTimeout(total=30)) as r:
        try:
            wait = int(float(r.headers.get("retry-after", 0)))
        except ValueError:
            wait = 0
        data = await r.json(content_type=None)
        if r.status != 200:
            log.warning("%s/%s -> %s %s", p, model, r.status, str(data)[:160])
            return r.status, None, wait
    try:
        if p == "gemini":
            return 200, data["candidates"][0]["content"]["parts"][0]["text"], 0
        return 200, data["choices"][0]["message"]["content"], 0
    except Exception:
        return 204, None, 0


async def llm(session, prompt):
    PRIORITY = ["groq", "gemini", "openrouter"]
    keys = {"groq": GROQ_KEY, "gemini": GEMINI_KEY, "openrouter": OPENROUTER_KEY}
    usable = [p for p in PRIORITY if keys[p]]
    order = [p for p in usable if cooldown.get(p, 0) < time.time()]
    if not order:                       # everything exhausted -> go back to the 1st and try again
        log.warning("All LLM providers on cooldown, retrying from the first")
        cooldown.clear(); order = usable
    for p in order:
        if not models.get(p) or time.time() - models_at.get(p, 0) > 6 * 3600:
            await discover(session, p)
        for model in models.get(p, [])[:4]:
            try:
                status, text, wait = await call(session, p, model, prompt)
            except Exception as e:
                log.warning("%s/%s failed: %r", p, model, e)
                cooldown[p] = time.time() + 60; break
            if status == 200 and text:
                if models[p][0] != model:
                    models[p].remove(model); models[p].insert(0, model)   # remember what works
                return text
            if status == 429 or status >= 500:
                cooldown[p] = time.time() + min(max(60, wait + 1), 900); break   # out of quota -> next provider
            if status in (401, 403):
                cooldown[p] = time.time() + 1800; break
            # 404 / 400 / empty answer -> try the next model of this provider
        else:
            models_at[p] = 0            # nothing worked here -> rediscover models next time
            cooldown[p] = time.time() + 120
    return None


VOCAB_LIST = []                        # real-word filter only used by the emergency fallback; wordfreq (~200 MB RAM) is opt-in
VOCAB = None
if os.getenv("USE_WORDFREQ", "0") == "1":
    try:
        from wordfreq import top_n_list
        VOCAB_LIST = [w for w in top_n_list("en", int(os.getenv("VOCAB_WORDS", "150000"))) if w.isascii() and w.isalpha() and len(w) >= 2]
        VOCAB = set(VOCAB_LIST)
        log.info("Vocabulary loaded: %d words", len(VOCAB))
    except Exception as _e:
        log.warning("wordfreq not available (%r)", _e)

# ---------------- solver loading ----------------
EMB = {"model": None, "state": "not started"}


def ensure_solver():
    """start loading the word-vector solver exactly once (safe to call many times)"""
    if EMB["state"] == "not started":
        EMB["state"] = "loading"
        asyncio.get_running_loop().run_in_executor(None, load_solver)


def load_solver():
    """runs in a worker thread: loads the precomputed word-vector files that ship with the bot (a couple of seconds)"""
    try:
        import solver as solver_mod
        t = time.time()
        m = solver_mod.build(None)
        import gc; gc.collect()
        m.set_history(S.answers)
        EMB["model"] = m
        EMB["state"] = "ready"
        log.info("=== SOLVER READY: %d candidate words, loaded in %.1fs, %d remembered answers, memory %.0f MB ===",
                 len(m.words), time.time() - t, len(S.answers), _rss_mb())
    except Exception as e:
        EMB["state"] = f"failed: {e!r}"
        log.error("=== SOLVER NOT LOADED: %r === (guesses will be weak until this is fixed; run selfcheck.py)", e)

# ---------------- daily claim DM ----------------
from datetime import datetime, timedelta, timezone
IST = timezone(timedelta(hours=5, minutes=30))
CLAIM_BOT = os.getenv("CLAIM_BOT", "IrisceneBot").lstrip("@")
CLAIM_TEXT = os.getenv("CLAIM_TEXT", ".claim")
CLAIM_HOUR = int(os.getenv("CLAIM_HOUR", "1"))           # 1 = 01:00 IST
CLAIM_MIN = int(os.getenv("CLAIM_MIN", "0"))
claim_state = {"last": "never", "count": 0}


def seconds_until_claim(now=None):
    now = now or datetime.now(IST)
    target = now.replace(hour=CLAIM_HOUR, minute=CLAIM_MIN, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds(), target


async def daily_claim():
    while True:
        wait, target = seconds_until_claim()
        log.info("Next %s to @%s at %s IST (in %.1f h)", CLAIM_TEXT, CLAIM_BOT, target.strftime("%Y-%m-%d %H:%M"), wait / 3600)
        await asyncio.sleep(wait)
        for attempt in range(1, 6):                       # retry up to 5 times if something goes wrong
            try:
                await client.send_message(CLAIM_BOT, CLAIM_TEXT)
                claim_state["count"] += 1
                claim_state["last"] = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
                log.info("Sent %s to @%s", CLAIM_TEXT, CLAIM_BOT)
                break
            except errors.FloodWaitError as e:
                await asyncio.sleep(e.seconds + 2)
            except Exception as e:
                log.warning("claim attempt %d failed: %r", attempt, e)
                await asyncio.sleep(60)
        await asyncio.sleep(90)                           # make sure we don't fire twice for the same minute


@client.on(events.NewMessage(incoming=True))
async def on_dm(event):
    try:
        if event.is_private:
            sender = await event.get_sender()
            if (getattr(sender, "username", "") or "").lower() == CLAIM_BOT.lower():
                log.info("@%s replied: %s", CLAIM_BOT, (event.raw_text or "").replace("\n", " | ")[:200])
    except Exception:
        pass



# ---------------- Iris messages ----------------
@client.on(events.NewMessage(chats=GROUP))
@client.on(events.MessageEdited(chats=GROUP))
async def on_msg(event):
    try:
        text = event.raw_text or ""
        low = text.lower()
        if S.active and PLAIN_WORD.match(text.strip()) and not isinstance(event, getattr(events.MessageEdited, "Event", ())):
            S.tried.add(text.strip().lower())              # a word someone typed in the group = a guess made by another player: "taken"
        if not any(k.lower() in low for k in MARKERS):
            return                                         # not a Trace-related message: ignore without any API call
        sender = await event.get_sender()
        if not getattr(sender, "bot", False):
            return
        _events()
        S.last_iris = time.time()
        S.trace_streak = 0                                 # Iris is alive and answering
        log.info("IRIS: %s", text.replace("\n", " | ")[:220])

        if COMPLETE_RE.search(text):
            w = WINNER_RE.search(text); a = ANSWER_RE.search(text)
            winner = w.group(1).strip() if w else "?"
            mine = bool(ME["name"]) and ME["name"].lower() in winner.lower()
            answer = a.group(1).lower() if a else None
            if event.message.id not in S.counted:
                S.counted.add(event.message.id)
                if mine:
                    S.my_wins += 1
                remember_answer(answer)
                try:
                    with open("rounds.jsonl", "a", encoding="utf8") as fh:
                        fh.write(json.dumps({"answer": answer, "guesses": S.round_guesses, "ranks": dict(S.known)}) + "\n")
                except Exception:
                    pass
                log.info("ROUNDDATA %s", json.dumps({"answer": answer, "ranks": dict(sorted(S.known.items(), key=lambda kv: kv[1])[:40])}))
                log.info("TRACE COMPLETE | answer=%s winner=%s %s | my guesses=%d | my wins=%d (~%d aura)",
                         answer, winner, "(ME!)" if mine else "(someone else)", S.round_guesses, S.my_wins, S.my_wins * 500)
                finish_round("trace complete", count=True)
            return

        if "Group session started" in text:
            if event.message.id == S.session_msg_id and S.active:     # Iris EDITS this message during a round: ignore
                S.ack_evt.set(); return
            if S.active and S.round_guesses > 0 and time.time() - S.last_gameplay < 15:
                log.warning("ignored a 'session started' message while the round is clearly running (kept %d tried words)", len(S.tried))
                S.ack_evt.set(); return
            S.session_msg_id = event.message.id
            reset_round(); log.info("New round started")
            S.ack_evt.set(); return
        if "Trace is ongoing" in text:
            if not S.active:
                S.active, S.started, S.round_guesses = True, time.time(), 0
                log.info("Joined a running round")
            if time.time() - S.just_ended < 90:             # we asked to end it and it is still running
                S.abandon_blocked = True
                log.warning("'.trace end' did not end the round (only the host/admin can) - I keep playing it")
            S.ack_evt.set(); return
        gameplay = ("Current guess" in text) or ("Top 10 closest" in text)
        if EXPIRED_RE.search(text) or ((not gameplay) and ("TRACE" in text.upper()) and ENDED_RE.search(text)):
            finish_round(text.replace("\n", " ")[:60], count=True)
            S.ack_evt.set(); return

        if TAKEN_RE.search(text) or NOTWORD_RE.search(text):
            rid = getattr(event.message, "reply_to_msg_id", None)
            w = S.ids.get(rid) if rid else None
            w = w or S.pending
            if w:
                S.tried.add(w)
                log.info("rejected by Iris: %s (%s)", w, "taken" if TAKEN_RE.search(text) else "not a word")
            if w == S.pending or not rid:
                S.reply_evt.set()
            return

        cur = CURRENT.search(text)
        cw = cr = None
        if cur:
            cw, cr = cur.group(1).lower(), int(cur.group(3))
        else:
            c2 = CURRENT2.search(text)
            if c2:
                cw, cr = c2.group(1).lower(), int(c2.group(2))
        if "Hints unlocked" in text:
            hs = RANK_LINE.findall(text.split("Hints unlocked", 1)[1])
            if hs:
                log.info("HINT from Iris: %s", ", ".join(f"{w.lower()} #{r}" for r, w in hs))
                for r, w in hs:
                    w = w.lower(); S.known[w] = min(int(r), S.known.get(w, 10 ** 9)); S.tried.add(w)
        if "Top 10 closest" in text or cw:
            S.last_gameplay = time.time()
            if not S.active and time.time() - S.finished_at > 20:    # ignore late edits of a round that just ended
                S.active, S.started = True, time.time()
            if cw:
                S.known[cw] = min(cr, S.known.get(cw, 10 ** 9)); S.tried.add(cw)
            for r, w in RANK_LINE.findall(text):
                w = w.lower(); S.known[w] = min(int(r), S.known.get(w, 10 ** 9)); S.tried.add(w)
            if cw and cw == S.pending:
                S.reply_evt.set()
            S.ack_evt.set()
    except Exception:
        log.exception("handler error")


def finish_round(reason, count=True):
    S.why_inactive = reason
    if S.active and count:
        S.games += 1
        log.info("ROUND OVER (%s) | rounds finished: %d", reason, S.games)
    S.active = False
    S.finished_at = time.time()
    if S.reply_evt is not None:
        S.reply_evt.set(); S.ack_evt.set()


# ---------------- Telegram actions ----------------
async def say(text):
    while True:
        try:
            return await client.send_message(GROUP, text)
        except errors.FloodWaitError as e:
            S.extra = min(S.extra + 2.0, 20.0)
            log.warning("FloodWait %ss - waiting it out, guess gap now +%.1fs", e.seconds, S.extra)
            await asyncio.sleep(e.seconds + 2)
        except errors.SlowModeWaitError as e:
            await asyncio.sleep(e.seconds + 1)


async def wait_ack(seconds):
    _events(); S.ack_evt.clear()
    try:
        await asyncio.wait_for(S.ack_evt.wait(), seconds)
        return True
    except asyncio.TimeoutError:
        return False


async def resync_from_history():
    """Look at the last messages in the group. True (and S.active/S.known filled) if a Trace round is clearly still running.
    This is what stops a restart or a missed message from causing a needless '.trace'."""
    try:
        msgs = await client.get_messages(GROUP, limit=40)
    except Exception as e:
        log.warning("could not read chat history: %r", e); return False
    now = datetime.now(timezone.utc)
    known, active, first_date, mine, sess_id = {}, None, None, set(), None
    for m in msgs:                                           # newest first
        text = m.raw_text or ""
        low = text.lower()
        if getattr(m, "out", False) and text.strip().isalpha() and active is not False:
            mine.add(text.strip().lower())                   # a word WE sent earlier (kept so a restart never repeats it)
            continue
        if not any(k.lower() in low for k in MARKERS):
            continue
        try:
            if (now - m.date).total_seconds() > ROUND_SECONDS:
                break
            snd = m.sender or await m.get_sender()
        except Exception:
            continue
        if not getattr(snd, "bot", False):
            continue
        gameplay = ("Current guess" in text) or ("Top 10 closest" in text)
        if COMPLETE_RE.search(text) or EXPIRED_RE.search(text) or ((not gameplay) and "TRACE" in text.upper() and ENDED_RE.search(text)):
            if active is None:
                active = False
            break
        if "Group session started" in text:
            if active is None:
                active = True
            first_date = m.date; sess_id = m.id
            break
        if gameplay or "Trace is ongoing" in text:
            if active is None:
                active = True
            first_date = m.date
            c = CURRENT.search(text) or CURRENT2.search(text)
            if c:
                w = c.group(1).lower(); r = int(c.group(c.lastindex))
                known[w] = min(r, known.get(w, 10 ** 9))
            for r, w in RANK_LINE.findall(text):
                known[w.lower()] = min(int(r), known.get(w.lower(), 10 ** 9))
    if active:
        S.active = True
        S.started = first_date.timestamp() if first_date else time.time()
        S.last_iris = time.time()
        for w, r in known.items():
            S.known[w] = min(r, S.known.get(w, 10 ** 9)); S.tried.add(w)
        S.tried |= mine
        if sess_id:
            S.session_msg_id = sess_id
        log.info("Round already running (found in chat history, %d words known, %d of my own words) - NOT sending .trace", len(known), len(mine))
        if time.time() - S.just_ended < 90:                  # we tried to end it and it is still there
            S.abandon_blocked = True
    return bool(active)


async def start_round():
    if await resync_from_history():
        S.last_trace = time.time(); S.trace_streak = 0
        return
    need = TRACE_GAP if S.trace_streak == 0 else min(TRACE_BACKOFF_MAX, TRACE_BACKOFF_BASE * 2 ** (S.trace_streak - 1))
    wait = need - (time.time() - S.last_trace)
    if wait > 0:
        if S.trace_streak:
            log.info("Iris did not answer the last %d .trace - waiting %.0fs before trying again", S.trace_streak, wait)
        t_end = time.time() + wait
        while time.time() < t_end and not S.paused and not S.active:     # 1 s steps: pause / timer commands stay responsive
            await asyncio.sleep(1)
        if S.paused or S.active:
            return
    S.last_trace = time.time(); S.trace_sends += 1
    _events(); S.ack_evt.clear()
    log.info("Sending .trace #%d (unanswered in a row: %d; previous round ended because: %s)", S.trace_sends, S.trace_streak, S.why_inactive)
    await say(".trace")
    await wait_ack(12)
    if S.active:
        S.trace_streak = 0; S.trace_fail = 0
        return
    S.trace_streak += 1
    S.trace_fail += 1
    if S.trace_fail >= 2:                                  # Iris stays silent: assume a round is running, keep guessing
        log.info("No answer to .trace twice - assuming a round is running and guessing")
        S.active, S.started, S.last_iris, S.round_guesses = True, time.time(), time.time(), 0
        S.trace_fail = 0


def best_rank():
    return min([r for r in S.known.values() if r < 100000] or [10 ** 9])


def should_abandon():
    """user rule: round older than STALE_SECONDS and not a single word better than STALE_RANK -> end it"""
    if not S.active or S.abandon_blocked:
        return None
    el = time.time() - S.started
    if el > STALE_SECONDS and S.round_guesses >= STALE_MIN_GUESSES and best_rank() > STALE_RANK:
        b = best_rank()
        return f"{int(el)}s and best rank is {('#%d' % b) if b < 10 ** 9 else 'none'} (needs better than #{STALE_RANK})"
    return None


async def abandon_round(reason):
    log.warning("ABANDONING this round: %s -> '.trace end' and starting a fresh one", reason)
    S.just_ended = time.time()
    await say(".trace end")
    await wait_ack(10)
    finish_round("abandoned", count=True)
    S.last_trace = time.time() - TRACE_GAP / 2                # next .trace may go out soon


# ---------------- choosing the next word ----------------
def valid_word(w):
    return VOCAB is None or w in VOCAB


async def fallback_words(http, n):
    """solver not available: opening seeds, then the free LLMs, then random common words"""
    if len(S.tried) < 10:
        pool = [w for w in SEEDS if w not in S.tried and valid_word(w)]
        random.shuffle(pool)
        return pool[:n]
    best = sorted(S.known.items(), key=lambda kv: kv[1])[:20]
    words = []
    for _ in range(2 if LEGACY_LLM else 0):
        prompt = PROMPT.format(
            best="\n".join(f"{r}: {w}" for w, r in best) or "none yet",
            far=", ".join([w for w, r in S.known.items() if r >= 100000][-30:]) or "none",
            tried=", ".join(sorted(S.tried | set(words))[-250:]), n=n + 4)
        txt = await llm(http, prompt)
        if not txt:
            break
        found = re.findall(r'"([A-Za-z]{2,20})"', txt) or re.findall(r"[A-Za-z]{2,20}", txt)
        for w in found:
            w = w.lower()
            if w not in S.tried and w not in words and valid_word(w):
                words.append(w)
        if len(words) >= n:
            break
    if words:
        log.info("GUESS [llm] %s", ", ".join(words[:n]))
        return words[:n]
    pool = [w for w in VOCAB_LIST[:15000] if w not in S.tried] or [w for w in SEEDS if w not in S.tried]
    w = random.sample(pool, min(n, len(pool))) if pool else []
    log.warning("GUESS [EMERGENCY/random - solver not active] %s", ", ".join(w))
    return w


def stem(w):
    """crude stem: ships/ship, tried/try, playing/play ... Iris answers variants of a word almost identically, so they waste a guess"""
    for suf, rep in (("ies", "y"), ("ied", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", ""), ("ly", ""), ("er", "")):
        if w.endswith(suf) and len(w) - len(suf) + len(rep) >= 3:
            w = w[: len(w) - len(suf)] + rep
            break
    if len(w) > 3 and w[-1] == w[-2]:
        w = w[:-1]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


def is_variant(w, stems):
    return stem(w) in stems


async def _llm_job(http, known, tried, rid):
    try:
        hints = []
        m = EMB["model"]
        if m is not None and known:
            try:
                hints = m.recent_top(tried, 12)                  # no new computation: reuse the last scoring
            except Exception:
                hints = []
        words = await cfai.propose(http, known, tried, 12, hints=hints)
    except Exception:
        log.exception("LLM job failed"); return
    if words and rid == S.round_id:
        S.llm_words = [w for w in words if w not in S.tried]


def maybe_start_llm(http):
    """background request: never delays a guess; its answer is used from the next guess on"""
    if cfai.blocked():
        return
    if S.llm_task is not None and not S.llm_task.done():
        return
    b = best_rank()
    improved = b < 10 ** 9 and b < S.llm_best / 2                      # a clearly better clue just arrived: ask again right away
    if S.llm_words and S.round_guesses - S.llm_last < LLM_EVERY and not improved:
        return
    S.llm_last = S.round_guesses; S.llm_best = b
    S.llm_task = asyncio.ensure_future(_llm_job(http, dict(S.known), set(S.tried), S.round_id))


async def pick_word(http):
    """solver pick, optionally replaced by the best LLM proposal (judged by the same word-vector score)"""
    if EMB["model"] is None and EMB["state"] in ("not started", "loading"):
        ensure_solver()
        return None                                          # solver still loading: wait, do not waste guesses
    maybe_start_llm(http)
    m = EMB["model"]
    loop = asyncio.get_running_loop()
    sol = None
    if m is not None:
        try:
            stems = {stem(t) for t in S.tried}
            c = await loop.run_in_executor(None, m.suggest, dict(S.known), set(S.tried), 6)
            c = [w for w in c if w not in S.tried and not is_variant(w, stems)]
            sol = c[0] if c else None
        except Exception:
            log.exception("solver error - falling back")
    stems = {stem(t) for t in S.tried}
    cands = [w for w in S.llm_words if w not in S.tried and not is_variant(w, stems)]
    if cands:
        informative = best_rank() <= (m.maxrank if m is not None else 30000)
        if informative and m is not None:
            sc = await loop.run_in_executor(None, m.score_words, dict(S.known), cands + ([sol] if sol else []))
            top = max(cands, key=lambda w: sc[w])
            if sol is None or sc[top] >= sc[sol] - LLM_MARGIN:
                S.llm_words.remove(top)
                log.info("GUESS [llm] %s (score %.1f vs solver %s)", top, sc[top], ("%s %.1f" % (sol, sc[sol])) if sol else "-")
                return top
        elif not informative:
            w = cands[0]; S.llm_words.remove(w)
            log.info("GUESS [llm probe] %s", w)
            return w
    if sol:
        return sol
    w = await fallback_words(http, 1)
    return w[0] if w else None


async def send_guess(word):
    _events()
    S.reply_evt.clear()
    S.pending = word
    S.tried.add(word)
    msg = await say(word)
    try:
        S.ids[msg.id] = word
    except Exception:
        pass
    S.sent += 1; S.last_sent = time.time()
    S.round_guesses += 1
    if S.sent % 40 == 0 and S.extra > 0:                   # 40 sends without trouble -> speed up a little
        S.extra = max(0.0, S.extra - 0.5)
    try:
        await asyncio.wait_for(S.reply_evt.wait(), REPLY_WAIT)
    except asyncio.TimeoutError:
        log.info("no answer from Iris for '%s' within %.0fs", word, REPLY_WAIT)
    S.pending = None
    if S.round_guesses % 5 == 0 or S.round_guesses <= 3:
        log.info("guess #%d '%s' | known=%d best=%s", S.round_guesses, word, len(S.known),
                 ("#%d" % best_rank()) if best_rank() < 10 ** 9 else "-")


# ---------------- controls: pause / resume / status ----------------
def status_text():
    el = int(time.time() - S.started) if S.active else 0
    b = best_rank()
    return (f"v{VERSION} up {int((time.time() - S.boot) // 60)}m | {'PAUSED' if S.paused else 'RUNNING'} | in a round: {S.active}"
            + (f" ({el // 60}m{el % 60:02d}s, {S.round_guesses} guesses, best {('#%d' % b) if b < 10 ** 9 else 'none'})" if S.active else "")
            + f" | solver: {EMB['state']} | rounds finished: {S.games} | my wins: {S.my_wins} (~{S.my_wins * 500} aura)"
            + f" | LLM: {'off (no CF keys)' if not cfai.enabled else ('cooling down' if cfai.blocked() else 'on')} ok={cfai.state['ok']}/{cfai.state['calls']}"
            + f" | memory {_rss_now():.0f} MB"
            + f" | .trace sent: {S.trace_sends}"
            + f" | answers remembered: {len(S.answers)} | guess delay: {SEND_DELAY + S.extra:.1f}s")


def timer_text():
    parts = []
    if S.timer_rounds is not None:
        parts.append(f"after {max(0, S.timer_rounds - S.games)} more round(s)")
    if S.timer_until:
        left = max(0, int(S.timer_until - time.time()))
        parts.append(f"in {left // 3600}h{(left % 3600) // 60:02d}m")
    return ("STOP TIMER: " + " / ".join(parts)) if parts else "no stop timer"


def set_timer(rounds=None, hours=None):
    if rounds:
        S.timer_rounds = S.games + int(rounds)
    if hours:
        S.timer_until = time.time() + float(hours) * 3600


def clear_timer():
    S.timer_rounds = None; S.timer_until = None


async def notify(text):
    try:
        await client.send_message("me", text)
    except Exception as e:
        log.warning("could not notify: %r", e)


def timer_due():
    if S.timer_until and time.time() >= S.timer_until:
        return "time limit reached"
    if S.timer_rounds is not None and S.games >= S.timer_rounds:
        return "round limit reached"
    return None


def apply_cmd(cmd, arg=None):
    if cmd == "pause":
        if not S.paused:
            S.paused, S.paused_at = True, time.time()
        msg = "Paused. Nothing will be sent until you send 'resume'. " + status_text()
    elif cmd == "resume":
        S.paused = False
        msg = "Resumed. " + status_text()
    elif cmd == "timer" and arg:
        set_timer(rounds=arg.get("rounds"), hours=arg.get("hours"))
        msg = "Timer set. " + timer_text() + ". The bot will PAUSE itself then (send 'resume' to continue). " + status_text()
    elif cmd == "timer_off":
        clear_timer(); msg = "Timer cancelled. " + status_text()
    else:
        msg = status_text() + " | " + timer_text()
    log.info("CONTROL: %s %s -> %s", cmd, arg or "", msg[:70])
    return msg


CMD_RE = re.compile(r"^\s*[/!.]?\s*(pause|resume|status|stop|start|continue)\s*[.!]?\s*$", re.I)
CMD_MAP = {"pause": "pause", "stop": "pause", "resume": "resume", "start": "resume", "continue": "resume", "status": "status"}
TIMER_RE = re.compile(r"^\s*[/!.]?\s*(?:stop after|pause after|disable after|off after|timer(?: set)?)\s+(\d+(?:\.\d+)?)\s*"
                      r"(rounds?|games?|hours?|hrs?|h|minutes?|mins?|m)\s*[.!]?\s*$", re.I)
TIMER_OFF_RE = re.compile(r"^\s*[/!.]?\s*(?:timer\s+(?:off|cancel|stop|clear)|cancel\s+timer|no\s+timer)\s*[.!]?\s*$", re.I)
TIMER_SHOW_RE = re.compile(r"^\s*[/!.]?\s*timer\s*[.!?]?\s*$", re.I)


def parse_command(text):
    """text -> (cmd, arg) or None.  pause | resume | status | timer {rounds|hours} | timer_off"""
    text = text or ""
    m = CMD_RE.match(text)
    if m:
        return CMD_MAP[m.group(1).lower()], None
    if TIMER_OFF_RE.match(text):
        return "timer_off", None
    if TIMER_SHOW_RE.match(text):
        return "status", None
    m = TIMER_RE.match(text)
    if m:
        n = float(m.group(1)); u = m.group(2).lower()
        if u.startswith(("round", "game")):
            return "timer", {"rounds": max(1, int(n))}
        if u.startswith(("m",)) and not u.startswith("mo"):
            return "timer", {"hours": n / 60.0}
        return "timer", {"hours": n}
    return None


@client.on(events.NewMessage())
async def on_cmd(event):
    try:
        if not event.is_private:
            return
        parsed = parse_command(event.raw_text)
        if not parsed:
            return
        cmd, arg = parsed
        saved = bool(event.out and ME["id"] and event.chat_id == ME["id"])      # typed in this account's own Saved Messages
        owner = bool(event.sender_id in OWNERS and not event.out)               # any of the OWNER_IDS writing to this account
        if not (saved or owner):
            log.info("command '%s' ignored: sent by id=%s in chat=%s out=%s. Use the Saved Messages of the BOT account "
                     "(%s, id %s) or add %s to OWNER_IDS", cmd, event.sender_id, event.chat_id, event.out,
                     ME["name"], ME["id"], event.sender_id)
            return
        reply = apply_cmd(cmd, arg)
        await client.send_message(event.chat_id, reply)
    except Exception:
        log.exception("command handler error")


async def h_health(_):
    b = best_rank()
    return web.Response(text=f"ok | {status_text()} | last_claim={claim_state['last']}")


async def h_cmd(request):
    cmd = request.match_info["cmd"]
    if not CONTROL_KEY:
        return web.Response(status=403, text="Controls are off: set the CONTROL_KEY environment variable first.")
    if request.query.get("key") != CONTROL_KEY:
        return web.Response(status=403, text="wrong key")
    q = request.query
    if cmd == "timer":
        arg = {}
        if q.get("rounds"):
            arg["rounds"] = int(float(q["rounds"]))
        if q.get("hours"):
            arg["hours"] = float(q["hours"])
        if q.get("minutes"):
            arg["hours"] = float(q["minutes"]) / 60.0
        if not arg:
            return web.Response(text="use /timer?key=..&rounds=5  or  &hours=10  or  &minutes=90")
        return web.Response(text=apply_cmd("timer", arg))
    if cmd == "timeroff":
        return web.Response(text=apply_cmd("timer_off"))
    if cmd not in ("pause", "resume", "status"):
        return web.Response(status=404, text="unknown command")
    return web.Response(text=apply_cmd(cmd))


async def h_panel(request):
    """a page with big Pause / Resume / Status buttons: https://YOUR-APP.onrender.com/panel?key=CONTROL_KEY"""
    if not CONTROL_KEY:
        return web.Response(status=403, text="Controls are off: add an environment variable CONTROL_KEY (any password) on Render, then open /panel?key=THAT_PASSWORD")
    if request.query.get("key") != CONTROL_KEY:
        return web.Response(status=403, text="wrong key")
    cmd = request.query.get("do", "")
    q = request.query
    if cmd == "timer":
        arg = {}
        if q.get("rounds"):
            arg["rounds"] = int(float(q["rounds"]))
        if q.get("hours"):
            arg["hours"] = float(q["hours"])
        note = apply_cmd("timer", arg) if arg else status_text()
    elif cmd in ("pause", "resume", "status", "timer_off"):
        note = apply_cmd(cmd)
    else:
        note = status_text() + " | " + timer_text()
    k = CONTROL_KEY
    html = ("<html><head><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<style>body{font-family:sans-serif;margin:20px}a{display:block;margin:14px 0;padding:22px;text-align:center;"
            "font-size:24px;border-radius:12px;color:#fff;text-decoration:none}</style></head><body>"
            f"<p><b>{note}</b></p>"
            f"<a style='background:#d33' href='/panel?key={k}&do=pause'>PAUSE</a>"
            f"<a style='background:#2a2' href='/panel?key={k}&do=resume'>RESUME</a>"
            f"<a style='background:#36c' href='/panel?key={k}&do=status'>STATUS / REFRESH</a>"
            f"<a style='background:#a60' href='/panel?key={k}&do=timer&rounds=5'>STOP AFTER 5 ROUNDS</a>"
            f"<a style='background:#a60' href='/panel?key={k}&do=timer&rounds=10'>STOP AFTER 10 ROUNDS</a>"
            f"<a style='background:#a60' href='/panel?key={k}&do=timer&hours=2'>STOP AFTER 2 HOURS</a>"
            f"<a style='background:#a60' href='/panel?key={k}&do=timer&hours=10'>STOP AFTER 10 HOURS</a>"
            f"<a style='background:#666' href='/panel?key={k}&do=timer_off'>CANCEL TIMER</a>"
            f"<a style='background:#555' href='/rounds?key={k}'>DOWNLOAD rounds.jsonl</a>"
            f"<a style='background:#555' href='/debug?key={k}'>DEBUG: last log lines</a></body></html>")
    return web.Response(text=html, content_type="text/html")


async def h_rounds(request):
    """download what the bot has learned: /rounds?key=CONTROL_KEY (rounds.jsonl) and /answers?key=CONTROL_KEY"""
    if not CONTROL_KEY or request.query.get("key") != CONTROL_KEY:
        return web.Response(status=403, text="set CONTROL_KEY on Render and open /rounds?key=THAT_PASSWORD")
    name = "answers.json" if request.path.rstrip("/").endswith("answers") else "rounds.jsonl"
    try:
        with open(name, encoding="utf8") as fh:
            body = fh.read()
    except FileNotFoundError:
        body = f"({name} does not exist yet - it is created when a round finishes; Render also wipes it on every restart/redeploy)"
    return web.Response(text=body, content_type="text/plain")


async def h_debug(request):
    if not CONTROL_KEY or request.query.get("key") != CONTROL_KEY:
        return web.Response(status=403, text="set CONTROL_KEY on Render and open /debug?key=THAT_PASSWORD")
    txt = status_text() + "\n" + f"solver state: {EMB['state']} | tried words this round: {len(S.tried)} | known ranks: {len(S.known)}\n\n" + "\n".join(list(RING.buf)[-200:])
    return web.Response(text=txt, content_type="text/plain")


async def watchdog():
    """warns (and finally re-syncs) when a round is 'active' but nothing has been sent for a long time"""
    while True:
        await asyncio.sleep(20)
        try:
            if S.active and not S.paused and (EMB["model"] is not None or EMB["state"].startswith("failed")):
                idle = time.time() - max(S.last_sent, S.started)
                if idle > 90:
                    log.warning("STALL: round active but no guess sent for %ds | solver=%s paused=%s last Iris message %ds ago",
                                idle, EMB["state"], S.paused, time.time() - S.last_iris)
                if idle > 240:
                    S.active = False; S.why_inactive = "watchdog: nothing sent for 4 min"; S.last_sent = time.time()
        except Exception:
            log.exception("watchdog error")


def start_console():
    """type pause / resume / status in the window that runs the bot (only when started from a terminal)"""
    if not sys.stdin or not sys.stdin.isatty():
        return
    loop = asyncio.get_running_loop()

    def reader():
        while True:
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if not line:
                return
            pc = parse_command(line)
            if pc:
                loop.call_soon_threadsafe(lambda c=pc: print(">>", apply_cmd(*c), flush=True))
            elif line.strip():
                print(">> type: pause | resume | status | stop after 5 rounds | stop after 10 hours | timer off", flush=True)
    threading.Thread(target=reader, daemon=True).start()
    log.info("Console controls on: type pause / resume / status here and press Enter")


# ---------------- main loop ----------------
async def play():
    ensure_solver()
    await client.start()
    if not S.dialogs_loaded:                               # only once per run, repeated calls trigger flood waits
        await client.get_dialogs(limit=200)                # fills Telethon's entity cache so the group id resolves
        S.dialogs_loaded = True
    try:
        grp = await client.get_entity(GROUP)
        log.info("Group found: %s", getattr(grp, "title", GROUP))
    except Exception as e:
        log.error("Cannot open group %s (is this account a member? right id?): %r", GROUP, e)
        raise
    me = await client.get_me()
    ME["name"], ME["id"], ME["username"] = me.first_name or "", me.id, me.username or ""
    log.info("Logged in as %s (id %s). Controls: send pause / resume / status in THIS account's Saved Messages%s",
             ME["name"], ME["id"], f", or from owners {sorted(OWNERS)}" if OWNERS else "")
    _events()
    nxt = None
    if EMB["state"] == "loading":
        log.info("Waiting for the solver to finish loading (can take a minute on a small server)...")
        for _ in range(240):
            if EMB["state"] != "loading":
                break
            await asyncio.sleep(0.5)
    if EMB["model"] is None:
        log.error("!!! SOLVER IS NOT ACTIVE (%s). Guesses will be weak/random. Check that emb_glove.npz, emb_fasttext.npz and solver.py "
                  "are next to bot.py, and that the server has enough memory (look for 'SOLVER' lines above).", EMB["state"])
    await resync_from_history()
    if STOP_AFTER_ROUNDS or STOP_AFTER_HOURS:
        set_timer(rounds=STOP_AFTER_ROUNDS or None, hours=STOP_AFTER_HOURS or None); log.info("Startup %s", timer_text())
    async with aiohttp.ClientSession() as http:
        while True:
            if S.paused:
                log.info("PAUSED - waiting for 'resume'")
                while S.paused:
                    S.last_iris = time.time()
                    await asyncio.sleep(1)
                if time.time() - S.paused_at > 240:        # long pause: the round may be gone -> re-sync from chat history
                    S.active = False; S.why_inactive = "long pause"
                log.info("RESUMED")
                continue
            if time.time() - S.last_memlog > 300:
                S.last_memlog = time.time(); log.info("memory now %.0f MB (Render free limit is 512 MB)", _rss_now())
            due = timer_due()
            if due and not S.paused:
                clear_timer()
                S.paused, S.paused_at = True, time.time()
                log.warning("STOP TIMER: %s - bot is now PAUSED (send 'resume' to continue)", due)
                await notify(f"⏹ Timer: {due}. The bot paused itself. Send 'resume' to continue. " + status_text())
                continue
            why = should_abandon()
            if why:
                await abandon_round(why); continue
            now = time.time()
            if S.active and now - S.started > ROUND_SECONDS + 60:
                S.active = False; S.why_inactive = "round older than 46 min"
            if S.active and now - S.last_iris > 420:       # Iris silent for 7 min while we play -> re-sync
                log.info("Iris silent for 7 min - re-syncing")
                S.active = False; S.why_inactive = "Iris silent for 7 min"
            if not S.active:
                if S.games and S.games % PAUSE_EVERY == 0 and S.rested_at_games != S.games:
                    S.rested_at_games = S.games
                    log.info("%d rounds finished -> resting %ds", S.games, PAUSE_SECONDS)
                    for _ in range(PAUSE_SECONDS):
                        if S.paused:
                            break
                        await asyncio.sleep(1)
                    continue
                await start_round(); continue
            if nxt is not None and (nxt[1] != S.round_id or nxt[0] in S.tried):
                nxt = None                                      # the word chosen in advance is no longer valid
            word = nxt[0] if nxt else await pick_word(http)
            nxt = None
            if not word:
                await asyncio.sleep(2); continue
            if word in S.tried:                                 # last line of defence against repeats
                log.warning("BLOCKED duplicate word '%s'", word); await asyncio.sleep(0.5); continue
            t0 = time.time()
            await send_guess(word)
            if S.active and not S.paused:                       # choose the NEXT word now, while the gap timer runs
                try:
                    w2 = await pick_word(http)
                    if w2:
                        nxt = (w2, S.round_id)
                except Exception:
                    log.exception("pre-pick failed")
            wait = SEND_DELAY + S.extra + random.random() * 0.3 - (time.time() - t0)
            if wait > 0:
                await asyncio.sleep(wait)


def _rss_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return -1


async def main():
    log.info("=== Trace bot v%s starting (memory %.0f MB) ===", VERSION, _rss_mb())
    load_answers()
    ensure_solver()
    app = web.Application()
    app.add_routes([web.get("/", h_health), web.get("/panel", h_panel), web.get("/rounds", h_rounds), web.get("/debug", h_debug), web.get("/answers", h_rounds), web.get("/{cmd}", h_cmd)])
    runner = web.AppRunner(app); await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", "7860"))).start()
    asyncio.create_task(watchdog())
    asyncio.create_task(daily_claim())                     # runs forever next to the game loop
    start_console()
    while True:                                            # never die, just restart on error
        try:
            await play()
        except Exception:
            log.exception("crashed, restarting in 20s"); await asyncio.sleep(20)

if __name__ == "__main__":
    asyncio.run(main())
