"""Run:  python selfcheck.py   (in the same folder as bot.py). It tells you in plain words if the solver can work."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ok = True
def bad(msg):
    global ok; ok = False; print("  [PROBLEM]", msg)
print("1) libraries")
try:
    import numpy; print("  numpy", numpy.__version__)
except Exception as e: bad(f"numpy missing -> run: pip install numpy   ({e!r})")
print("  (wordfreq is no longer needed - it used ~200 MB of RAM)")
try:
    import telethon, aiohttp; print("  telethon", telethon.__version__, "| aiohttp ok")
except Exception as e: bad(f"telethon/aiohttp missing -> run: pip install telethon aiohttp   ({e!r})")
print("2) files in", HERE)
for f in ["bot.py", "solver.py", "cfai.py", "emb_glove.npz", "emb_fasttext.npz"]:
    p = os.path.join(HERE, f)
    if os.path.exists(p): print(f"  {f:18s} {os.path.getsize(p)/1e6:6.1f} MB")
    else: bad(f"{f} is MISSING from this folder")
for f in []:
    p = os.path.join(HERE, f)
    if os.path.exists(p): print(f"  {f:18s} {os.path.getsize(p)/1e6:6.1f} MB  (rare words)")
    else: print(f"  {f} not found - rare words will be missing (copy it from the zip for best results)")
if ok:
    print("3) building the solver (takes a few seconds)")
    try:
        import solver
        t = time.time(); E = solver.build(None); print(f"  built in {time.time()-t:.1f}s, {len(E.words)} candidate words")
        obs = {"partnership":15,"cooperation":23,"participation":26,"concern":36,"involvement":52,"contribution":104,"agreement":140}
        out = E.suggest(obs, set(obs), 10)
        print("  demo 1: ranks for 'partnership, cooperation, participation...' -> bot guesses:", out)
        print("          (the real answer was 'commitment')", "-> FOUND" if "commitment" in out else "-> not in first 10 here")
        junk = {w: 100001 for w in "higher admin bot sorry your".split()}
        print("  demo 2: no useful clue yet -> bot explores everyday concepts:", E.suggest(junk, set(junk), 5))
    except Exception as e:
        bad(f"solver failed to build: {e!r}")
print("4) Cloudflare AI (free LLM helper)")
try:
    import asyncio, aiohttp, cfai
    if not cfai.enabled:
        print("  CF_ACCOUNT_ID / CF_API_TOKEN not set here -> the LLM helper is OFF (the bot still works with the solver alone)")
    else:
        async def _t():
            async with aiohttp.ClientSession() as h:
                w = await cfai.propose(h, {"ship": 84, "boat": 120, "sea": 300, "door": 9000, "hospital": 20000}, {"ship", "boat", "sea"}, 8)
                print("  LLM proposals for a ship/boat/sea clue:", w or "NONE", "| model used:", cfai.state["last_ok"], "| last error:", cfai.state["last_err"] or "-")
                if not w: bad("Cloudflare AI gave no answer - check the key names/permissions (see last error above)")
        asyncio.run(_t())
except Exception as e:
    print("  could not test Cloudflare AI:", repr(e))
print()
print("ALL GOOD - the solver will be used." if ok else "FIX THE [PROBLEM] LINES ABOVE, then run this again.")
