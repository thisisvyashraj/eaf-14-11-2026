# Trace bot v3

## Secrets / environment variables (Render -> Environment)
| name | needed | what |
|---|---|---|
| API_ID, API_HASH, SESSION_STRING, GROUP | yes | same as before |
| CF_ACCOUNT_ID | for the LLM | Cloudflare dashboard -> Workers AI (or the right side of any Workers page) -> Account ID |
| CF_API_TOKEN | for the LLM | Cloudflare -> My Profile -> API Tokens -> Create Token -> template "Workers AI" |
| CONTROL_KEY | for web control | any password you invent, e.g. `abc123xyz` |
| OWNER_ID | optional | numeric Telegram id of your MAIN account (so you can send pause/resume from it) |
| CF_MODELS | optional | comma list of Cloudflare model ids, best first (defaults are set) |
| SEND_DELAY | optional | seconds between guesses (default 4; lower = faster but risks FloodWait) |
| STALE_SECONDS / STALE_RANK | optional | abandon rule (default 300 s / 5000) |
| PAUSE_EVERY / PAUSE_SECONDS | optional | rest after N finished rounds (default 8 / 180 s) |

GROQ_KEY / GEMINI_KEY / OPENROUTER_KEY still work as an old fallback but are no longer needed.

## Control
* Telegram: in the **Saved Messages of the bot account (Moonknight)** send `pause`, `resume` or `status` (the bot answers there).
  From your main account: set OWNER_ID, then message the bot account privately.
* Web: `https://YOUR-APP.onrender.com/pause?key=CONTROL_KEY` (also `/resume`, `/status`). Plain `/` shows the status.
* Console (when you run `python bot.py` on a PC): type pause / resume / status + Enter.
If a command is ignored the log says exactly why.

## Files
`rounds.jsonl` gets every finished round with ALL ranks seen (send it to improve the solver). `answers.json` = remembered answers.
Run `python selfcheck.py` first; it also tests the Cloudflare key.

## Solver (v3.1): how it finds the word
Every rank Iris returns - even a bad one like #12000 - says how far the secret is from that guess. The solver turns each
rank into a target similarity and scores every candidate word by how well it matches ALL targets at once
(trilateration), plus a 30% share of the old "average of the closest words" score. On the two rounds I had full logs for
(tennis, craft) the true word moved from ~16th-31st place to ~1st-5th place. It then guesses the top candidate every time
(no more random exploring once 3 ranks are known).

Optional tuning (environment variables, leave alone unless needed):
`EMB_NT` (45000, assumed size of Iris' vocabulary; 35000-50000 all worked), `EMB_CW` (0.3), `EMB_TPW` (1.0), `EMB_TRI=0` (switch trilateration off),
`EMB_FEXP` (0, share of guesses spent on exploring after the opening).
Memory: the solver peaks around 380 MB. If Render kills the service for memory, set `EMB_NT=35000` and/or remove the
`wordfreq` line from requirements.txt (the bot then loses only the "typical word" prior).

## v3.2 - reading the log / what to check first
* Startup shows `=== Trace bot v3.2 starting ===`. If you see that line again and again, the service is crashing and restarting
  (usually out of memory on a 512 MB server) - set `EMB_NT=35000`, remove `wordfreq` from requirements.txt, or use a bigger plan.
* You must see `=== SOLVER READY ... ===`. If you see `SOLVER NOT LOADED` / `SOLVER IS NOT ACTIVE`, the .npz files or solver.py are missing
  next to bot.py, or memory ran out. The bot now WAITS for the solver instead of guessing random words.
* Before sending `.trace` the bot reads the last chat messages; if Iris' round is still running it just joins it (no `.trace`).
  Every `.trace` that is sent logs why ("previous round ended because: ...").
* The old Groq/Gemini/OpenRouter helpers are OFF by default (their quotas are used up and they only filled the log). Set `LEGACY_LLM=1` to re-enable.

## v3.3 - memory fix (why v3.2 kept restarting)
v3.2 used 480+ MB right after loading and Render's 512 MB limit killed it ("Instance restarted"). v3.3 drops `wordfreq` (~200 MB),
and the 217k rare-word files (not needed: Iris' vocabulary is ~50k common words). The solver now needs ~290 MB at most.
`emb_glove_x1.npz` / `emb_glove_x2.npz` can be deleted from your repo (set `EMB_EXT=emb_glove_x1.npz,emb_glove_x2.npz` only if you
have a bigger server and want rare words). `USE_WORDFREQ=1` re-enables the old word-frequency prior (needs `wordfreq` in requirements.txt and ~200 MB more RAM).

## v3.4 - controls that are hard to get wrong
Easiest: set `CONTROL_KEY` on Render (any password, e.g. `mypass123`), then open
`https://YOUR-APP.onrender.com/panel?key=mypass123` on your phone - big PAUSE / RESUME / STATUS buttons.
Telegram way: send `pause` / `resume` / `status` in the Saved Messages of the BOT account (Moonknight) - nowhere else.
Anywhere else the log shows: `command 'pause' ignored: sent by id=...`.

## v3.5 - repeated words fix, several owners, stop timer
* Repeats: Iris EDITS its "Group session started" message during a round; v3.4 treated every edit as a new round and forgot the
  words already tried. Fixed (edits are ignored). After a restart the bot also re-reads ITS OWN earlier words from the chat.
* Owners: `OWNER_IDS=111111,222222` (comma separated numeric Telegram ids). Each owner writes to the BOT account (Moonknight) in a private chat.
  Anyone not listed is ignored (the log shows their id so you can add it). Saved Messages of Moonknight always works too.
* Stop timer (the bot PAUSES itself, `resume` continues). Send any of these to Moonknight (Saved Messages / owner DM), type it in the console, or use the /panel buttons:
  `stop after 5 rounds`  |  `stop after 10 hours`  |  `stop after 90 min`  |  `timer` (shows what is left)  |  `timer off`
  When the timer fires you get a message in Moonknight's Saved Messages.
* Start with a timer already set: env `STOP_AFTER_ROUNDS=5` and/or `STOP_AFTER_HOURS=10`.
* Web: `/panel?key=CONTROL_KEY` has buttons for all of it (also `/timer?key=..&rounds=5`, `&hours=10`, `&minutes=90`, `/timeroff?key=..`).

## v3.6 - more repeat protection, faster, smarter helper
* Words that OTHER players type in the group are now also marked as "taken" (Iris says "already taken" for any word anyone used).
* Variants of a tried word (ship/ships, try/tried, play/playing) are skipped: Iris answers them almost the same, so they only waste a guess.
* A second "Group session started" message while a round is clearly running is ignored (the log says "ignored a 'session started' message").
* Guess gap default is now 3.0 s (+0-1 s random) instead of 4 (env SEND_DELAY). The bot slows itself down automatically after a FloodWait.
* The LLM now also sees the solver's top candidates, and is asked again immediately when a clearly better clue arrives (not only every 3 guesses).
* Finding from your old logs: rank #100001 does NOT mean "unknown word" - the same word is #100001 in one round and #7000 in another. It means "very far".
Still repeating? Send me the 10 log lines around one repeat (look for "rejected by Iris" and "New round started").

## v3.7 - download rounds.jsonl
Open `https://YOUR-APP.onrender.com/rounds?key=CONTROL_KEY` (also a button on /panel) and save the page as rounds.jsonl
(or copy the text). `/answers?key=...` shows answers.json. Render wipes these files on every restart/redeploy, so download soon after a few rounds.
No CONTROL_KEY set? Then copy the `ROUNDDATA {...}` lines from the Render logs instead.

## v3.8 - smaller memory (Render "exceeded its memory limit" emails)
The solver now keeps only the 60,000 most common words per model and a 30,000-word working size: ~195 MB peak instead of ~290 MB
in my tests, with the same accuracy on the real rounds I have (craft, tennis). `status` and the log show the live memory
("memory now NNN MB" every 5 min). Render's free limit is 512 MB for EVERYTHING (Python + Telegram + solver).
Still restarting? In Render -> Environment add `EMB_FILES=emb_glove.npz` (one model instead of two: ~60 MB less) and, if needed,
`EMB_MAXW=40000`. Also add `MALLOC_ARENA_MAX=2` if it is not there. Send me the "memory now" lines before the restart.

## v4.0 - full strength inside Render's 512 MB, and much less CPU per guess
* Render's free instance has a tiny slice of CPU (your log: the solver took 46 s to load there, 1-2 s on a normal PC), so every recalculation per guess was slow.
  The solver now UPDATES its scores incrementally: a new guess costs one small matrix-vector product instead of redoing everything
  (~20 ms per guess on a PC, was ~60 ms+; same or better accuracy on the real rounds).
* Nothing is trimmed any more: all 100,000 most common words of BOTH word models, working size 45,000. Memory ~240 MB for the solver (peak in my test).
  The rare-word files are off by default (they made accuracy worse and need 2x the RAM).
* The next word is now chosen WHILE the gap timer runs, so solving time no longer adds to the gap. Defaults: SEND_DELAY=2.8 s (+0-0.3 s), REPLY_WAIT=5 s.
  10 guesses take about 30-35 s if Telegram allows it; after a FloodWait the bot slows itself down automatically.
* IMPORTANT: delete these from Render -> Environment if you added them earlier: EMB_FILES, EMB_MAXW, EMB_NT, EMB_TMAX, EMB_EXT (they would shrink it again).
  Keep MALLOC_ARENA_MAX=2 and SEND_DELAY (set it to 2.8 or delete it).

## v4.1 - how to be sure WHICH bot is running, and no more .trace spam
* The very first log line must be `=== Trace bot v4.1 starting ...`, and `https://YOUR-APP.onrender.com/` must show `v4.1`. If you see log lines like
  HEARTBEAT or trace_level, Render is running some OTHER bot.py (not this one): delete every old file in the repo and upload this whole folder again.
* `.trace` is now rate-limited: if Iris does not answer, the bot waits 60 s, then 120, 240 ... (max 10 min) instead of repeating it. The log says "Sending .trace #N".
* The 5-minute "abandon" rule only applies after at least 15 guesses (STALE_MIN_GUESSES), so a slow server cannot end rounds before it has tried.
* A watchdog logs `STALL: ...` if a round is active but nothing was sent for 90 s, and re-syncs after 4 min.
* `/debug?key=CONTROL_KEY` (also a button on /panel) shows the status and the last 200 log lines - copy them to me when something goes wrong.

## v5.0 - it learns (and the bugs your last log showed)
Bugs fixed, found in your log:
* "GUESS [EMERGENCY/random ...]" with an empty word = the solver had no word left to try (it only probed ~330 concepts while no clue was "near"). It now keeps
  probing with the ~4000 most common words, each as far as possible from everything tried, and never runs dry (tested with 1500 "everything is far" guesses).
* Iris' newer announcement ("WORD TRACE / Duration: 45min / Join - /trace ...") is now recognised as a new session (before, the bot did not notice it and sent `.trace` again).
* "Very far" ranks (#100001 or beyond the vocabulary) are no longer thrown away: they now tell the solver which regions NOT to look in
  (simulation with 73% capped ranks: words solved in <=30 guesses went from 8/30 to 17/30; no change when nothing is capped).
* Past answers no longer get a big bonus (they were being guessed first and were all wrong).
Self-improvement:
* After every finished round the bot stores the round (the ranks Iris gave + the answer). tuner.py then replays ALL stored rounds and re-picks a few settings
  (assumed vocabulary size, weight of the two word models, centroid weight, far-rank weight, word-frequency prior) so that the true answers rank as high as possible,
  also when only half or a quarter of the clues are used. Takes a few seconds, runs between rounds, and only changes settings if it helps by >= 5%.
* Everything it learns (rounds, answers, tuned settings) is saved as ONE file in the bot account's Saved Messages ("tracebot_state.json" - please do not delete it),
  because Render wipes its own disk on every restart. It is loaded again at start.
* `status` and /debug show how many rounds it has learned from and the last tuning result. It needs about 4 finished rounds (it starts with the 2 real rounds I have logs for).
Honest limits: it improves the settings, it cannot see inside Iris' word model, and a round whose answer is very far from everything is still hard.
