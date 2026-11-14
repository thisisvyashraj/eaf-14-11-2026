"""Self-tuning: learns Iris' behaviour from FINISHED rounds (answer known).
After every finished round the stored rounds are replayed and a few solver settings (assumed vocabulary size, weights of the
two word models, centroid weight, far-rank weight ...) are re-chosen to put the true answers as high as possible in the
candidate list - at full, half and a quarter of the clues, so early guesses count too. Cheap on purpose (candidate SAMPLE,
cached per-guess tables) because Render's CPU is tiny; works in small time slices and resumes later."""
import time, math, logging, zlib
import numpy as np

log = logging.getLogger("trace")
GRID = {
    "nt": [25000, 35000, 45000, 60000],
    "cw": [0.0, 0.15, 0.3, 0.6, 1.0],
    "tpw": [0.5, 1.0, 2.0],
    "far_w": [0.0, 0.1, 0.15, 0.25, 0.4],
    "cprior": [0.2, 0.5, 0.8, 1.2],
    "mw0": [0.5, 1.0, 2.0],
}
ORDER = ["nt", "far_w", "cw", "mw0", "tpw", "cprior"]
MAX_OBS = 40
MIN_OBS = 6
SUBSETS = (1.0, 0.5, 0.25)


class Tuner:
    def __init__(self, ens, sample=2000, max_rounds=30, seed=1):
        self.e = ens
        self.rng = np.random.RandomState(seed)
        self.sample = min(sample, len(ens.words))
        self.base = self.rng.choice(len(ens.words), self.sample, replace=False)
        self.max_rounds = max_rounds
        self.cache = {}                                       # round key -> prepared tables
        self.logfreq = np.log1p(ens.freq / 20000.0)

    # ---- per-round tables (the expensive part, done once per round) ----------------------------------------------------
    def _prep(self, rnd):
        e = self.e
        a = e.u.get(rnd["answer"])
        if a is None:
            return None
        items = [(w, int(r)) for w, r in rnd["ranks"].items()]
        if len(items) < MIN_OBS:
            return None
        if len(items) > MAX_OBS:
            items.sort(key=lambda kv: kv[1]); h = MAX_OBS // 2
            rest = items[h:]; step = max(1, len(rest) // (MAX_OBS - h)); items = items[:h] + rest[::step][:MAX_OBS - h]
        cand = np.unique(np.concatenate([self.base, [a]]))
        apos = int(np.where(cand == a)[0][0])
        rs = np.random.RandomState(zlib.crc32(rnd["answer"].encode()) % (2 ** 31))
        subs = []
        for f in SUBSETS:
            idx = np.arange(len(items)) if f >= 1.0 else np.sort(rs.choice(len(items), max(3, int(len(items) * f)), replace=False))
            subs.append(idx)
        tabs = []
        for mi, mod in enumerate(e.m):
            rows = e.u2r[mi][cand]
            have = rows >= 0
            obs = []
            for k, (w, r) in enumerate(items):
                i = mod.idx.get(w)
                if i is None:
                    continue
                s_full = mod.Xc @ mod.Xc[i]
                cos_c = np.full(len(cand), np.nan, np.float32)
                cos_c[have] = mod.Xc[rows[have]] @ mod.Xc[i]
                T = {}
                for nt in GRID["nt"]:
                    n = min(nt, len(s_full)); sl = s_full[:n]
                    if r < 100000 and r <= nt:
                        kk = int(r)
                        T[nt] = float(np.partition(sl, n - kk)[n - kk]) if kk < n else float(sl.min())
                    else:
                        T[nt] = float(np.partition(sl, n - max(1, int(n * 0.5)))[n - max(1, int(n * 0.5))])   # far-rank threshold (median)
                obs.append((k, r, cos_c, T, i))
            tabs.append({"rows": rows, "have": have, "obs": obs})
        # centroid part per subset (parameter free)
        cen = []
        for idx in subs:
            per_model = []
            for mi, mod in enumerate(e.m):
                tb = tabs[mi]
                inner = sorted([(r, i) for (k, r, c, T, i) in tb["obs"] if k in set(idx.tolist()) and r <= e.maxrank])[:e.max_obs]
                if not inner:
                    per_model.append(None); continue
                w = np.array([1.0 / max(r, 1) for r, _ in inner]); w /= w.sum()
                O = mod.Xc[[i for _, i in inner]]
                c = (w[:, None] * O).sum(0); c /= (np.linalg.norm(c) + 1e-9)
                s = np.full(len(cand), np.nan, np.float32)
                s[tb["have"]] = mod.Xc[tb["rows"][tb["have"]]] @ c
                v = s[tb["have"]]
                s = (s - v.mean()) / (v.std() + 1e-9)
                per_model.append(s)
            cen.append(per_model)
        return {"cand": cand, "apos": apos, "tabs": tabs, "subs": subs, "cen": cen, "lf": self.logfreq[cand]}

    # ---- evaluate one parameter set on the prepared rounds ---------------------------------------------------------------
    def _score_round(self, P, p):
        e = self.e
        cand = P["cand"]; n = len(cand)
        mw = [p["mw0"]] + [1.0] * (len(e.m) - 1)
        out = []
        for si, idx in enumerate(P["subs"]):
            idxset = set(idx.tolist())
            tri_tot = np.zeros(n); tri_cnt = np.zeros(n); cen_tot = np.zeros(n); cen_cnt = np.zeros(n)
            for mi, tb in enumerate(P["tabs"]):
                have = tb["have"]
                S = np.zeros(n); W = 0.0; used = 0
                for (k, r, cos_c, T, i) in tb["obs"]:
                    if k not in idxset:
                        continue
                    d = np.nan_to_num(cos_c - T[p["nt"]], nan=0.0)
                    if r >= 100000 or r > p["nt"]:
                        if p["far_w"] <= 0:
                            continue
                        d = np.maximum(d, 0.0); wt = p["far_w"]
                    else:
                        wt = 1.0 / (1.0 + math.log(r)) ** p["tpw"]
                    S += wt * d * d; W += wt; used += 1
                if used >= 2:
                    sc = -S / W
                    v = sc[have]
                    z = np.where(have, (sc - v.mean()) / (v.std() + 1e-9), 0.0)
                    tri_tot += mw[mi] * z * have; tri_cnt += mw[mi] * have
                cz = P["cen"][si][mi]
                if cz is not None:
                    ok = ~np.isnan(cz)
                    cen_tot += mw[mi] * np.where(ok, cz, 0.0); cen_cnt += mw[mi] * ok
            if tri_cnt.max() == 0 and cen_cnt.max() == 0:
                out.append(math.log(len(e.words) / 2)); continue
            tri = tri_tot / np.maximum(tri_cnt, 1e-9)
            tri = np.where(tri_cnt > 0, tri, tri[tri_cnt > 0].min() if (tri_cnt > 0).any() else 0.0)
            zt = (tri - tri.mean()) / (tri.std() + 1e-9)
            cen = cen_tot / np.maximum(cen_cnt, 1e-9)
            cz = np.where(cen_cnt > 0, (cen - cen[cen_cnt > 0].mean()) / (cen[cen_cnt > 0].std() + 1e-9), -9.0) if (cen_cnt > 0).any() else np.zeros(n)
            score = zt + p["cw"] * cz - p["cprior"] * P["lf"]
            a = P["apos"]
            pos = 1 + int((score > score[a]).sum()) * (len(e.words) / n)
            out.append(math.log(pos))
        return out

    def evaluate(self, p, keys):
        vals = []
        for k in keys:
            if k in self.cache and self.cache[k] is not None:
                vals += self._score_round(self.cache[k], p)
        return float(np.mean(vals)) if vals else None

    # ---- main entry: tune within a time budget -----------------------------------------------------------------------------
    def tune(self, rounds, cur, budget=40.0, min_rounds=4, min_gain=0.05):
        """rounds: [{'answer','ranks'}]; cur: current params {'nt','cw','tpw','far_w','cprior','mw0'}.
        returns (new_params or None, info dict). Safe to call repeatedly; unfinished preparation resumes next time."""
        t0 = time.time()
        rounds = [r for r in rounds if r.get("answer") and len(r.get("ranks", {})) >= MIN_OBS][-self.max_rounds:]
        keys = []
        for r in rounds:
            key = (r["answer"], len(r["ranks"]), min(r["ranks"].values()) if r["ranks"] else 0)
            keys.append(key)
            if key not in self.cache:
                if time.time() - t0 > budget * 0.6:
                    continue                                  # out of time: prepare the rest next time
                try:
                    self.cache[key] = self._prep(r)
                except Exception as ex:
                    log.warning("tuner: could not prepare round %s: %r", r.get("answer"), ex); self.cache[key] = None
        for old in [k for k in self.cache if k not in keys]:
            del self.cache[old]
        ready = [k for k in keys if self.cache.get(k) is not None]
        info = {"rounds": len(ready), "pending": len(keys) - len(ready)}
        if len(ready) < min_rounds:
            info["note"] = f"need {min_rounds} rounds with an answer (have {len(ready)})"
            return None, info
        best = dict(cur); best_v = self.evaluate(best, ready)
        base_v = best_v; info["base"] = math.exp(base_v)
        for name in ORDER:
            if time.time() - t0 > budget:
                info["timeout"] = True; break
            for val in GRID[name]:
                if val == best[name]:
                    continue
                cand = dict(best); cand[name] = val
                v = self.evaluate(cand, ready)
                if v is not None and v < best_v - 0.01:
                    best, best_v = cand, v
        info["tuned"] = math.exp(best_v)
        gain = (base_v - best_v) / max(base_v, 1e-9)
        info["gain"] = gain
        if gain >= min_gain and best != cur:
            return best, info
        return None, info
