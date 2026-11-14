"""Word-vector solver for Iris' Trace game.
Idea: Iris tells us how close each guessed word is to the secret. Words closest in meaning to the BEST-ranked
guesses are the likeliest answers. With no useful clue yet, explore broad everyday concepts instead."""
import os, logging, threading
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one math thread: far less memory, plenty fast for this job
import numpy as np

log = logging.getLogger("trace")
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = [f.strip() for f in os.getenv("EMB_FILES", "emb_glove.npz,emb_fasttext.npz").split(",") if f.strip()]
def _mem_limit_mb():
    """memory limit of this container/VM in MB (None = unknown / unlimited)"""
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v.isdigit() and int(v) < 1 << 50:
                return int(v) / 1048576
        except Exception:
            pass
    return None


# rare-word files (emb_glove_x1/x2.npz) are OFF: in tests they crowd the shortlist with obscure words (worse accuracy) and need ~2x the RAM.
EXT = [f.strip() for f in os.getenv("EMB_EXT", "").split(",") if f.strip()]     # still possible: EMB_EXT=emb_glove_x1.npz,emb_glove_x2.npz

CONCEPTS = """animal bird fish insect snake horse dog cat deer wolf bear monkey whale shark frog spider rabbit
tree flower grass forest plant fruit vegetable bread meat cheese drink coffee wine sugar salt rice egg
mountain river sea ocean desert island lake rain snow wind storm fire sun moon star sky cloud earth stone metal gold iron glass
house room door window kitchen city village street bridge road church school hospital castle farm garden market shop
car train ship plane boat bicycle engine machine tool hammer knife weapon sword gun
head hand eye heart blood bone skin hair face body doctor medicine disease
shirt dress shoe hat clothes cotton silk
king queen soldier war army peace law court crime police prison
money bank price trade business company work job boss factory
book paper letter story poem word language song music dance movie theater painting art
computer phone internet software data code robot science physics chemistry biology math number
game sport football tennis race team player winner ball
teacher student child baby mother father family friend enemy love hate fear joy anger
time day night morning year history future memory dream
idea thought truth lie power force energy speed light dark color red blue green white black
water air ground wall roof floor key lock box bag bottle cup plate table chair bed lamp clock mirror
travel journey adventure danger secret magic ghost god religion spirit""".split()


STOP = set("""the and for are but not you all any can had her was one our out day get has him his how man new now old
see two way who boy did its let put say she too use that with have this will your from they know want been good much
some time very when come here just like long make many more only over such take than them well were what where which
while would about after again also because before being below between both could does doing down during each even ever
every few further had has having here hers herself himself into itself might more most must myself nor off once other
ought ours ourselves own same should so some still then there these those through under until upon was we whom why
yet yours yourself yourselves their theirs though although whether either neither shall may much many thing things
something anything everything someone anyone everyone nothing nobody okay yes yeah please thanks thank sorry hello
sure maybe really actually basically thus hence therefore however whatever whenever whoever whichever whereas""".split())


def _read(path):
    d = np.load(path)
    words = d["w"].tobytes().decode("ascii").split("\n") if d["w"].dtype == np.uint8 else [str(x) for x in d["w"]]
    return words, d["V"]


class _Model:
    """One word-vector file. Every row is stored ONCE as a float32, centred on the mean and normalised to length 1
    ("Xc"). With that, the centred cosine between two words is a plain dot product, so one new guess costs one matrix-vector product."""
    CH = 20000

    def __init__(self, words, Q):
        self.words = words
        self.idx = {w: i for i, w in enumerate(words)}
        N, d = len(words), Q.shape[1]
        X = np.empty((N, d), np.float32)
        tot = np.zeros(d, np.float64)
        for a in range(0, N, self.CH):
            blk = Q[a:a + self.CH].astype(np.float32)
            blk /= (np.linalg.norm(blk, axis=1, keepdims=True) + 1e-9)
            X[a:a + self.CH] = blk
            tot += blk.sum(0, dtype=np.float64)
        mu = (tot / N).astype(np.float32)
        for a in range(0, N, self.CH):
            blk = X[a:a + self.CH]
            blk -= mu
            blk /= (np.linalg.norm(blk, axis=1, keepdims=True) + 1e-9)
        self.Xc = X

    def rows(self, ids):
        return self.Xc[list(ids)]


class Ensemble:
    def __init__(self, models, allowed=None):
        self.m = models
        pos = {}
        for mod in models:
            for i, w in enumerate(mod.words):
                if len(w) >= 3 and (allowed is None or w in allowed) and (w not in pos or i < pos[w]):
                    pos[w] = i
        self.words = sorted(pos, key=pos.get)
        self.freq = np.array([pos[w] for w in self.words], float)
        self.u = {w: i for i, w in enumerate(self.words)}
        self.m2u = [np.array([self.u.get(w, -1) for w in mod.words]) for mod in models]
        self.rows_ok = [np.nonzero(u >= 0)[0] for u in self.m2u]            # rows of each model that are candidates
        self.uu = [u[r] for u, r in zip(self.m2u, self.rows_ok)]            # ... and their index in the unified list
        self.max_obs = int(os.getenv("EMB_OBS", "10"))
        self.alpha = float(os.getenv("EMB_ALPHA", "1.0"))
        self.maxrank = int(os.getenv("EMB_MAXRANK", "30000"))          # worse than this = no information for the centroid part
        self.cprior = float(os.getenv("EMB_CPRIOR", "0.8"))           # mild preference for commoner words
        self.cmodels = [int(x) for x in os.getenv("EMB_CMODELS", "0,1").split(",") if x.strip().isdigit()]
        self.concepts = [w for w in dict.fromkeys(CONCEPTS) if w in self.m[0].idx]
        self.zp = float(os.getenv("EMB_ZP", "1.0")); self.zh = float(os.getenv("EMB_ZH", "1.5"))
        self.zlo = float(os.getenv("EMB_ZLO", "3.3")); self.zhi = float(os.getenv("EMB_ZHI", "5.5"))
        self.hist = np.zeros(0, dtype=np.int64); self.hist_bonus = float(os.getenv("EMB_HIST", "1.5"))
        self.nt = int(os.getenv("EMB_NT", "45000"))                  # assumed size of Iris' vocabulary (tuned on real rounds: 35k-50k best)
        self.tpw = float(os.getenv("EMB_TPW", "1.0"))               # how strongly close ranks outweigh far ranks in trilateration
        self.cw = float(os.getenv("EMB_CW", "0.3"))                 # weight of the centroid score next to trilateration
        self.use_tri = os.getenv("EMB_TRI", "1") != "0"
        self.zipf = None
        if os.getenv("USE_WORDFREQ", "0") == "1":                      # optional prior; costs ~200 MB of RAM, so OFF by default
            try:
                from wordfreq import zipf_frequency
                self.zipf = np.array([zipf_frequency(w, "en") for w in self.words], dtype=np.float32)
            except Exception as e:
                log.warning("wordfreq unavailable for the typical-word prior (%r)", e)
        prior = self.cprior * np.log1p(self.freq / 20000.0)            # computed once, not on every guess
        if self.zipf is not None and (self.zp or self.zh):
            prior = prior + self.zp * np.maximum(0.0, self.zlo - self.zipf) + self.zh * np.maximum(0.0, self.zipf - self.zhi)
        self.prior = prior
        self.stopmask = np.array([w in STOP for w in self.words])
        self.policy = os.getenv("EMB_POLICY", "tri")
        self._lock = threading.RLock()
        self._last = None
        self.new_round()

    def new_round(self):
        with self._lock:
            self._acc = 0.0
            self._st = None
            self._cen_key = None; self._cen = None

    def set_history(self, words):
        """past secret words: if Iris repeats one, it gets a bonus and is found almost immediately"""
        self.hist = np.array(sorted({self.u[w] for w in words if w in self.u}), dtype=np.int64)

    # ---- trilateration, updated INCREMENTALLY: every rank (even a far one) tells how far the secret is from that guess ----
    def _tri_sync(self, obs):
        """If Iris ranked guess g at r, the secret's cosine to g should equal the r-th largest cosine among g's neighbours.
        For each NEW guess we add (cos(candidate, g) - target)^2 to a running total over all candidates (one matrix-vector product)."""
        cur = {w: r for w, r in obs.items() if r < 100000}
        st = self._st
        if st is not None:
            ap = st["ap"]
            if any((w in ap and ap[w] != r) for w, r in cur.items()) or any(w not in cur for w in ap):
                st = None                                   # a rank changed / observations were reset: start over
        if st is None:
            st = {"ap": {}, "S": [np.zeros(len(m.words), np.float32) for m in self.m], "W": [0.0] * len(self.m), "n": [0] * len(self.m)}
        new = [(w, r) for w, r in cur.items() if w not in st["ap"]]
        for mi, mod in enumerate(self.m):
            if mi not in self.cmodels and len(self.m) > len(self.cmodels):
                continue
            nt = min(self.nt, len(mod.words))
            for w, r in new:
                i = mod.idx.get(w)
                if i is None or i >= nt or r > nt:
                    continue
                s = mod.Xc @ mod.Xc[i]
                k = int(r)
                t = float(np.partition(s[:nt], nt - k)[nt - k]) if k < nt else float(s[:nt].min())     # the k-th largest similarity
                wt = 1.0 / (1.0 + np.log(r)) ** self.tpw
                s -= t; s *= s; s *= wt
                st["S"][mi] += s; st["W"][mi] += wt; st["n"][mi] += 1
        for w, r in new:
            st["ap"][w] = r
        self._st = st
        return st

    def tri_scores(self, obs):
        st = self._tri_sync(obs)
        U = len(self.words)
        total = np.zeros(U); cnt = np.zeros(U)
        for mi, mod in enumerate(self.m):
            if st["n"][mi] < 2:
                continue
            nt = min(self.nt, len(mod.words))
            sc = -(st["S"][mi] / st["W"][mi])
            ref = sc[:nt]
            z = (sc - ref.mean()) / (ref.std() + 1e-9)
            total[self.uu[mi]] += z[self.rows_ok[mi]]; cnt[self.uu[mi]] += 1
        if cnt.max() == 0:
            return None
        sc = total / np.maximum(cnt, 1)
        valid = cnt > 0
        v = sc[valid]; k = max(0, int(0.01 * len(v)))
        sc[~valid] = np.partition(v, k)[k]                           # words outside the common vocabulary: mild penalty
        return sc

    # ---- the older "average of the closest words" score (cached while the 10 closest guesses stay the same) ---------
    def _centroid_part(self, obs):
        items = sorted(((w, r) for w, r in obs.items() if r <= self.maxrank), key=lambda kv: kv[1])[:self.max_obs]
        if not items:
            return None
        key = tuple(items)
        if key == self._cen_key:
            return self._cen
        U = len(self.words)
        total = np.zeros(U); cnt = np.zeros(U)
        for mi, mod in enumerate(self.m):
            if mi not in self.cmodels and len(self.m) > len(self.cmodels):
                continue
            its = [(w, r) for w, r in items if w in mod.idx]
            if not its:
                continue
            wts = np.array([1.0 / (max(r, 1) ** self.alpha) for _, r in its]); wts /= wts.sum()
            O = mod.Xc[[mod.idx[w] for w, _ in its]]
            c = (wts[:, None] * O).sum(0); c /= (np.linalg.norm(c) + 1e-9)
            s = mod.Xc @ c
            z = (s - s.mean()) / (s.std() + 1e-9)
            total[self.uu[mi]] += z[self.rows_ok[mi]]; cnt[self.uu[mi]] += 1
        if cnt.max() == 0:
            return None
        score = total / np.maximum(cnt, 1)
        score[cnt == 0] = -99
        self._cen_key, self._cen = key, score
        return score

    def centroid_scores(self, obs):
        """final candidate score = trilateration over ALL ranks + 0.3 x centroid of the closest words + mild priors.
        obs: ALL observed {word: rank} (the two parts filter by themselves). None = nothing usable yet."""
        with self._lock:
            cen = self._centroid_part(obs)
            tri = None
            if self.use_tri:
                try:
                    tri = self.tri_scores(obs)
                except Exception as e:
                    log.warning("trilateration skipped: %r", e)
            if cen is None and tri is None:
                return None
            if tri is None:
                score = cen.copy()
            else:
                zt = (tri - tri.mean()) / (tri.std() + 1e-9)
                if cen is None:
                    score = zt
                else:
                    v = cen > -90
                    zc = (cen - cen[v].mean()) / (cen[v].std() + 1e-9)
                    score = np.where(v, zt + self.cw * zc, -99.0)
            score = score - self.prior
            if len(self.hist):
                score[self.hist] += self.hist_bonus
            self._last = score
            return score

    def recent_top(self, tried, n):
        """top candidates of the most recent scoring (no new computation) - used as hints for the LLM"""
        sc = self._last
        if sc is None:
            return []
        out = []
        for i in self._top(sc, 200):
            w = self.words[i]
            if w in tried or self.stopmask[i]:
                continue
            out.append(w)
            if len(out) >= n:
                break
        return out

    def score_words(self, obs, words):
        """centroid z-score of specific words (e.g. LLM proposals) so they can be compared with the solver's own pick.
        Words missing from the vectors get -3 (Iris most likely does not know them either)."""
        sc = self.centroid_scores(obs)
        if sc is None:
            return {w: 0.0 for w in words}
        ok = sc > -90
        mu, sd = float(sc[ok].mean()), float(sc[ok].std()) + 1e-9
        return {w: ((float(sc[self.u[w]]) - mu) / sd if w in self.u and sc[self.u[w]] > -90 else -3.0) for w in words}

    @staticmethod
    def _top(sc, k):
        k = min(k, len(sc))
        top = np.argpartition(-sc, k - 1)[:k]
        return top[np.argsort(-sc[top])]

    def centroid_suggest(self, obs, tried, n, skip=()):
        sc = self.centroid_scores(obs)
        if sc is None:
            return []
        out = []
        for i in self._top(sc, 600):
            w = self.words[i]
            if w in tried or w in skip or self.stopmask[i]:
                continue
            out.append(w)
            if len(out) >= n:
                break
        return out

    # ---- no clue yet: explore broad everyday concepts, farthest from what was already tried -------------------
    def explore(self, n, tried):
        mod = self.m[0]
        pool = [w for w in self.concepts if w not in tried and w not in STOP]
        if not pool or n <= 0:
            return []
        P = mod.rows([mod.idx[w] for w in pool])
        tv = list(mod.rows([mod.idx[w] for w in tried if w in mod.idx])) if any(w in mod.idx for w in tried) else []
        if tv:
            mind = 1 - (P @ np.array(tv).T).max(1)
        else:
            c = P.mean(0); c /= np.linalg.norm(c); mind = P @ c            # start from the most 'central' concept
            mind = mind - mind.min()
        out = []
        for _ in range(min(n, len(pool))):
            j = int(np.argmax(mind)); out.append(pool[j])
            mind = np.minimum(mind, 1 - P @ P[j]); mind[j] = -9
        return out

    def suggest(self, obs, tried, n):
        """n guesses. n == 1 is the normal mode now: the bot re-plans after EVERY reply from Iris."""
        inf = {w: r for w, r in obs.items() if r <= self.maxrank}
        best = min(inf.values()) if inf else None
        if self.policy == "old":
            if best is None:
                f = 1.0                                 # nothing useful yet: explore
            elif best <= 5000:
                f = 0.0                                 # a real clue: dig into it
            elif best <= 15000:
                f = 0.4
            else:
                f = 0.6
            if inf and len(inf) <= 2 and best > 100:    # one or two weak clues can mislead: keep exploring too
                f = max(f, 0.5)
        else:
            # trilateration uses every rank, so the solver's own best guess is also informative: explore only at the start
            k_inf = sum(1 for r in obs.values() if r <= self.nt)
            if k_inf < int(os.getenv("EMB_KMIN", "3")):
                f = 1.0 if (best is None or best > 3000) else 0.0
            else:
                f = float(os.getenv("EMB_FEXP", "0.0")) if (best is not None and best > 5000) else 0.0
        with self._lock:
            self._acc += f * n                              # share of guesses that explore, carried between calls
            k_exp = min(n, int(self._acc)); self._acc -= k_exp
        out = self.centroid_suggest(obs, tried, n - k_exp) if k_exp < n else []
        if len(out) < n:
            out += self.explore(n - len(out), set(tried) | set(out))
        return out[:n]


def build(allowed=None):
    models = []
    for k, f in enumerate(FILES):
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            log.warning("missing vector file %s", f); continue
        words, Q = _read(p)
        mw = int(os.getenv("EMB_MAXW", "100000"))                   # keep only the most common words (saves RAM; answers are common words)
        if len(words) > mw:
            words, Q = words[:mw], Q[:mw].copy()
        if k == 0 and EXT:                              # glove: rare-word extension files (only on servers with plenty of RAM)
            parts = [Q]
            for e in EXT:
                pe = os.path.join(HERE, e)
                if os.path.exists(pe):
                    w2, Q2 = _read(pe); words = words + w2; parts.append(Q2); log.info("loaded %s (+%d rare words)", e, len(w2))
            Q = np.concatenate(parts)
        models.append(_Model(words, Q)); log.info("loaded %s (%d words)", f, len(words))
        del Q
    if not models:
        raise RuntimeError("no vector files found next to bot.py")
    return Ensemble(models, allowed)
