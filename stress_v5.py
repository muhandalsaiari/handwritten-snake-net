"""Stress v5: random tangled snakes vs the rule, then the network alone.
python stress_v5.py [states] [games] [seed]"""
import sys, pickle
from multiprocessing import Pool
import numpy as np
import v5_rule as R
from build_v4 import encode, forward
from play import play
mats = pickle.load(open("v5_net.pkl", "rb"))
def net(body, food): return int(np.argmax(forward(mats, encode(body, food))))
def odd(seed):
    r = np.random.default_rng([77, seed]); bad = n = 0; kinds = {}
    while n < 50:
        want = int(r.integers(1, 99)); b = [int(r.integers(100))]
        if r.random() < .3:  # a stretch of one loop, some of it shifted
            o = R.LOOPS[int(r.integers(64))]; s = int(r.integers(100)); b = [o[(s - i) % 100] for i in range(want)]
        else:
            while len(b) < want:
                c = [y for y in R.NL[b[-1]] if y not in b]
                if not c: break
                b.append(min(c, key=lambda y: (sum(z not in b for z in R.NL[y]) if r.random() < .8 else r.random())))
        fr = [c for c in range(100) if c not in b]
        if not fr: continue
        f = int(r.choice(fr)); n += 1
        stuck = all(c in b[:-1] for c in R.NL[b[0]]); kinds[stuck] = kinds.get(stuck, 0) + 1
        out = forward(mats, encode(b, f))
        if sorted(out.tolist()) != [0, 0, 0, 1] or int(np.argmax(out)) != R.decide(b, f): bad += 1
    return bad, n
def g(s): return play(net, np.random.default_rng([s[1], s[0]]))
if __name__ == "__main__":
    ns, ng, seed = (int(v) for v in sys.argv[1:4])
    with Pool(14) as p:
        r = p.map(odd, range(ns // 50)); print("odd states", sum(x[1] for x in r), "mismatches", sum(x[0] for x in r), flush=True)
        r = p.map(g, [(i, seed) for i in range(ng)], chunksize=4)
    e = {}
    for x in r: e[x[0]] = e.get(x[0], 0) + 1
    w = [x[2] for x in r if x[0] == "win"]
    print(ng, "games", e, "moves", round(np.mean(w), 1), "best", min(w), "worst", max(w))
