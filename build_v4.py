"""Hand-written Snake network, version 4: sees the whole body.
No training, no data, no loss, no gradients. Every weight is set here from
the design, and it must make exactly the moves of v4_rule.py (test_v4.py).

Input (300): head one-hot (100), food one-hot (100), age of every cell (100):
0 = free, 1 = tail ... L = head.
Every unit is ReLU(weights . previous layer + bias). All numbers stay whole,
so the network is exact. Output: 4 values, one is 1 (the move), others 0.

How it works, layer by layer:
  1-4    basic facts: which cells are body/tail, the length, the v3 network
         (copied in whole), is the body in loop order, does the loop ahead
         clear in time, which moves are legal.
  floods 64 layers. Five flood fills at once, one cell further per layer:
         from the head (to find the path to the food), and from each of the
         4 next cells (can the snake still reach its tail after that move,
         and how far away is it).
  trace  64 layers. Walks the shortest path back from the food, one cell per
         layer, numbering the cells.
  vbody  gives every cell the age it would have once the food is eaten.
  flood  64 more layers: from the food over that imagined body, can it reach
         its tail?
  end    picks the mode (v3 / loop / path / stall) and the move.

A flood fill as weights: R(x) = "reached". Each layer computes, per cell,
  U = ReLU(s + A*(passable-1)),  V = ReLU(s - 1 + A*(passable-1)),
  where s = R(x) + R of its neighbours. U - V is 1 if s >= 1 and the cell is
  passable, else 0: the new R. Counting R over the layers gives distances.
"""
import numpy as np
import scipy.sparse as sp

import build_weights as v3w
import v4_rule as R
from build_weights import CELLS, dist, neighbour

A = 512  # gate size, bigger than any number the network carries
D = R.DMAX
K = R.K
M = R.MARGIN
X = range(CELLS)
NB = [[neighbour(x, m) for m in range(4)] for x in X]
# FROM[x][m]: the cell c with neighbour(c, m) == x (the head that move m brings to x)
FROM = [[None] * 4 for _ in X]
for c in X:
    for m in range(4):
        if NB[c][m] is not None:
            FROM[NB[c][m]][m] = c
LOOPDIR = [R.loop_move(c) for c in X]


class E:
    """A linear expression over the units of one layer (plus a constant)."""
    __slots__ = ("t", "c", "k")

    def __init__(self, t, c=None, k=0):
        self.t, self.c, self.k = t, c or {}, k

    def __add__(self, o):
        if not isinstance(o, E):
            return E(self.t, dict(self.c), self.k + int(o) if float(o).is_integer() else self.k + float(o))
        assert o.t == self.t, "mixing layers"
        c = dict(self.c)
        for i, v in o.c.items():
            c[i] = c.get(i, 0) + v
        return E(self.t, c, self.k + o.k)

    __radd__ = __add__

    def __neg__(self):
        return E(self.t, {i: -v for i, v in self.c.items()}, -self.k)

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, s):
        s = int(s) if float(s).is_integer() else float(s)
        return E(self.t, {i: v * s for i, v in self.c.items()}, self.k * s)

    __rmul__ = __mul__


def flat(v):
    if isinstance(v, E):
        yield v
    elif isinstance(v, dict):
        for k in sorted(v):
            yield from flat(v[k])
    else:
        for w in v:
            yield from flat(w)


def rebuild(v, it):
    if isinstance(v, E):
        return next(it)
    if isinstance(v, dict):
        return {k: rebuild(v[k], it) for k in sorted(v)}
    return [rebuild(w, it) for w in v]


class Net:
    def __init__(self, n_in):
        self.n_in = n_in
        self.layers = []  # (rows, cols, vals, bias)
        self.t = 0

    def step(self, keep, new):
        """One layer. `keep` values are carried (ReLU(x) = x, they are all
        >= 0); `new` are new pre-activations. Returns both as new units."""
        plan = list(keep.items()) + list(new.items())
        rows, cols, vals, bias = [], [], [], []
        r = 0
        for _, v in plan:
            for e in flat(v):
                assert e.t == self.t, "expression from the wrong layer"
                for i, w in e.c.items():
                    if w:
                        rows.append(r)
                        cols.append(i)
                        vals.append(w)
                bias.append(e.k)
                r += 1
        self.layers.append((rows, cols, vals, bias))
        self.t += 1
        units = iter([E(self.t, {i: 1}) for i in range(r)])
        return {k: rebuild(v, units) for k, v in plan}


def zero(t):
    return E(t, {}, 0)


def flood_steps(net, st, specs):
    """Run D flood layers. specs: name -> (start list, passable list, keep acc).
    Returns (st, R_final, acc_total) where acc counts the layers a cell was
    reached in (distance = D + 1 - acc)."""
    Rv = {n: s for n, (s, _, _) in specs.items()}
    acc = {n: None for n in specs}
    passable = {n: p for n, (_, p, _) in specs.items()}
    for k in range(1, D + 1):
        new = {}
        for n, (_, _, want) in specs.items():
            Rn, P = Rv[n], passable[n]
            s = [Rn[x] + sum(Rn[y] for y in NB[x] if y is not None) for x in X]
            new["U_" + n] = [s[x] + A * (P[x] - 1) for x in X]
            new["V_" + n] = [s[x] - 1 + A * (P[x] - 1) for x in X]
            new["P_" + n] = P
            if want and k >= 2:
                new["acc_" + n] = Rn if acc[n] is None else [acc[n][x] + Rn[x] for x in X]
        out = net.step(st, new)
        st = {key: out[key] for key in st}
        for n, (_, _, want) in specs.items():
            Rv[n] = [out["U_" + n][x] - out["V_" + n][x] for x in X]
            passable[n] = out["P_" + n]
            if want and k >= 2:
                acc[n] = out["acc_" + n]
    total = {n: ([acc[n][x] + Rv[n][x] for x in X] if acc[n] is not None else None) for n in specs}
    return st, Rv, total


def build():
    net = Net(3 * CELLS)
    H = [E(0, {x: 1}) for x in X]
    F = [E(0, {CELLS + x: 1}) for x in X]
    a = [E(0, {2 * CELLS + x: 1}) for x in X]

    # ---- 1: thresholds of the ages
    o = net.step({"H": H, "F": F, "a": a},
                 {"a1": [v - 1 for v in a], "a2": [v - 2 for v in a], "a3": [v - 3 for v in a]})
    a, a1, a2, a3 = o["a"], o["a1"], o["a2"], o["a3"]
    occ = [a[x] - a1[x] for x in X]          # cell is body
    blk = [a1[x] - a2[x] for x in X]         # age >= 2: blocked even after the tail moves
    eq2 = [blk[x] - (a2[x] - a3[x]) for x in X]
    T = [occ[x] - blk[x] for x in X]         # tail
    st = {"H": o["H"], "F": o["F"], "a": a, "occ": occ, "blk": blk, "eq2": eq2, "T": T}
    L = sum(occ)

    # ---- 2: length one-hot, loop-ahead check, order check (part 1)
    new = {
        "P": [L - k + 1 for k in range(0, 103)],
        "viol": [a[x] - dist(h, x) + M + A * (st["H"][h] - 1) + A * (occ[x] - 1)
                 for h in X for x in X if x != h],
        "eqp": {(x, m): [a[NB[x][m]] - a[x], a[NB[x][m]] - a[x] - 1, a[NB[x][m]] - a[x] - 2]
                for x in X for m in range(4) if NB[x][m] is not None},
    }
    o = net.step(st, new)
    st = {k: o[k] for k in st}
    P = o["P"]
    LN = [P[k] - 2 * P[k + 1] + P[k + 2] for k in range(1, 101)]   # LN[l-1] = [L == l]
    st["geK"] = P[K] - P[K + 1]
    viol = sum(o["viol"])
    eq = {key: u[0] - 2 * u[1] + u[2] for key, u in o["eqp"].items()}

    # ---- 3: v3 layer 1, loop-ahead result, order pairs, legal moves, eating
    W1, b1, W2, b2, W3 = v3w.build()
    v3_in = st["H"] + st["T"] + st["F"] + LN
    W1c = sp.csc_matrix(W1)
    h1 = []
    for j in range(W1.shape[1]):
        lo, hi = W1c.indptr[j], W1c.indptr[j + 1]
        e = zero(net.t) + float(b1[j])
        for i, w in zip(W1c.indices[lo:hi], W1c.data[lo:hi]):
            e = e + v3_in[i] * float(w)
        h1.append(e)
    S = [[st["H"][FROM[x][m]] if FROM[x][m] is not None else None for x in X] for m in range(4)]
    new = {
        "lok": [1 - viol],
        "e": {key: eq[key] + st["occ"][key[0]] - 1 for key in eq},
        "h1": h1,
        "legal": [[S[m][x] - st["blk"][x] for x in X if S[m][x] is not None] for m in range(4)],
        "eat": [[S[m][x] + st["F"][x] - 1 for x in X if S[m][x] is not None] for m in range(4)],
    }
    o = net.step(st, new)
    st = {k: o[k] for k in st}
    st["lok"] = o["lok"][0]
    ordS = sum(o["e"][(x, m)] * dist(x, NB[x][m]) for (x, m) in o["e"])
    st["legal"] = [sum(o["legal"][m]) for m in range(4)]
    eat = [sum(o["eat"][m]) for m in range(4)]
    h1 = o["h1"]

    # ---- 4: v3 layer 2, order result, what each move would leave behind
    W2c = sp.csc_matrix(W2)
    h2 = []
    for k in range(W2.shape[1]):
        lo, hi = W2c.indptr[k], W2c.indptr[k + 1]
        e = zero(net.t) + float(b2[k])
        for i, w in zip(W2c.indices[lo:hi], W2c.data[lo:hi]):
            e = e + h1[i] * float(w)
        h2.append(e)
    Lc = sum(st["occ"])
    new = {
        "ord": [100 - ordS, 99 - ordS],
        "h2": h2,
        "pu": [[st["eq2"][x] - eat[m] for x in X] for m in range(4)],
        "tb2": [[st["T"][x] + eat[m] - 1 for x in X] for m in range(4)],
        "small2": [[3 - Lc - eat[m], 2 - Lc - eat[m]] for m in range(4)],
    }
    o = net.step(st, new)
    st = {k: o[k] for k in st}
    st["ord"] = o["ord"][0] - o["ord"][1]
    st["v3log"] = [sum(o["h2"][k] * float(W3[k, m]) for k in range(W3.shape[0]) if W3[k, m]) for m in range(4)]
    st["small2"] = [o["small2"][m][0] - o["small2"][m][1] for m in range(4)]
    st["Tb"] = [[o["pu"][m][x] + o["tb2"][m][x] for x in X] for m in range(4)]
    pm = [[(1 - st["blk"][x]) + o["pu"][m][x] for x in X] for m in range(4)]
    for drop in ("eq2", "T"):
        del st[drop]

    # ---- floods: path from the head, and tail checks for each move
    t = net.t
    specs = {"path": (st["H"], [1 - st["blk"][x] for x in X], True)}
    for m in range(4):
        start = [st["H"][FROM[x][m]] if FROM[x][m] is not None else zero(t) for x in X]
        specs[f"m{m}"] = (start, pm[m], True)
    del st["blk"]
    st, Rf, acc = flood_steps(net, st, specs)

    # ---- T1: distances -> path steps; stall results
    accP = acc["path"]
    new = {
        "eqn": {(x, m): [accP[x] - accP[NB[x][m]] + 2, accP[x] - accP[NB[x][m]] + 1, accP[x] - accP[NB[x][m]]]
                for x in X for m in range(4) if NB[x][m] is not None},
        "dge": [[D - accP[x], D - 1 - accP[x]] for x in X],
        "d1": [[accP[x] - D + 1, accP[x] - D] for x in X],
        "accF": [accP[x] - A * (1 - st["F"][x]) for x in X],
        "accT": [[acc[f"m{m}"][x] - A * (1 - st["Tb"][m][x]) for x in X] for m in range(4)],
        "reach": [[st["Tb"][m][x] + Rf[f"m{m}"][x] - 1 for x in X] for m in range(4)],
    }
    o = net.step(st, new)
    st = {k: o[k] for k in st}
    eqn = {key: u[0] - 2 * u[1] + u[2] for key, u in o["eqn"].items()}
    dge2 = [o["dge"][x][0] - o["dge"][x][1] for x in X]
    st["d1"] = [o["d1"][x][0] - o["d1"][x][1] for x in X]
    accF = sum(o["accF"])
    accT = [sum(o["accT"][m]) for m in range(4)]
    reach = [sum(o["reach"][m]) for m in range(4)]
    del st["Tb"]

    # ---- T2
    new = {
        "eqm": {key: eqn[key] + dge2[key[0]] - 1 for key in eqn},
        "rF": [accF, accF - 1],
        "rand": [reach[m] + st["small2"][m] - 1 for m in range(4)],
        "sca": [accT[m] - A * st["small2"][m] for m in range(4)],
        "reach": reach,
        "accF": accF,
    }
    o = net.step(st, new)
    st = {k: o[k] for k in st}
    eqm = o["eqm"]
    st["pathok"] = o["rF"][0] - o["rF"][1]
    st["accF"] = o["accF"]
    rok = [o["reach"][m] + st["small2"][m] - o["rand"][m] for m in range(4)]
    score = [o["sca"][m] + (D + 1) * st["small2"][m] for m in range(4)]

    # ---- T3: first path step out of each cell; start the trace at the food
    def eqm_or0(x, m):
        return eqm.get((x, m))
    nxt_new = {}
    for (x, m) in eqm:
        e = eqm[(x, m)]
        for m2 in range(m):
            if (x, m2) in eqm:
                e = e - eqm[(x, m2)]
        nxt_new[(x, m)] = e
    new = {
        "nxt": nxt_new,
        "Q": [st["F"][x] + st["pathok"] - 1 for x in X],
        "safe": [st["legal"][m] + rok[m] - 1 for m in range(4)],
        "score": score,
    }
    o = net.step(st, new)
    st = {k: o[k] for k in st}
    for key in ("safe", "score"):
        st[key] = o[key]
    nxt = o["nxt"]
    Q = o["Q"]

    # ---- trace: walk the path back from the food, numbering its cells
    ON = J = None
    for i in range(D):
        new = {"nxt": nxt,
               "w": {key: Q[key[0]] + nxt[key] - 1 for key in nxt},
               "ON": Q if ON is None else [ON[x] + Q[x] for x in X],
               "J": [Q[x] * i for x in X] if J is None else [J[x] + Q[x] * i for x in X]}
        o = net.step(st, new)
        st = {k: o[k] for k in st}
        nxt, ON, J = o["nxt"], o["ON"], o["J"]
        Qn = [zero(net.t) for _ in X]
        for (x, m), w in o["w"].items():
            Qn[NB[x][m]] = Qn[NB[x][m]] + w
        Q = Qn
    ON = [ON[x] + Q[x] for x in X]
    J = [J[x] + Q[x] * D for x in X]

    # ---- vbody: ages once the food is eaten; first step of the path
    Lc = sum(st["occ"])
    new = {
        "von": [Lc + 1 - J[x] - A * (1 - ON[x]) for x in X],
        "vold": [st["a"][x] - D + st["accF"] - A * ON[x] for x in X],
        "FS": [ON[x] + st["d1"][x] - 1 for x in X],
    }
    o = net.step(st, new)
    st = {k: o[k] for k in st}
    vb = [o["von"][x] + o["vold"][x] for x in X]
    FS = o["FS"]
    for drop in ("d1", "accF", "a"):
        del st[drop]

    new = {
        "vb": [[vb[x], vb[x] - 1, vb[x] - 2] for x in X],
        "PF": [[st["H"][c] + FS[NB[c][m]] - 1 for c in X if NB[c][m] is not None] for m in range(4)],
    }
    o = net.step(st, new)
    st = {k: o[k] for k in st}
    v = o["vb"]
    vpass = [1 - (v[x][1] - v[x][2]) for x in X]
    st["vtail"] = [v[x][0] - 2 * v[x][1] + v[x][2] for x in X]
    st["PF"] = [sum(o["PF"][m]) for m in range(4)]

    # ---- flood over the imagined body: can the head (at the food) reach the tail?
    st, Rv, _ = flood_steps(net, st, {"v": (st["F"], vpass, False)})

    # ---- end: modes and the move
    Lc = sum(st["occ"])
    new = {
        "vr": [st["vtail"][x] + Rv["v"][x] - 1 for x in X],
        "vsm": [2 - Lc, 1 - Lc],
    }
    o = net.step({k: v for k, v in st.items() if k not in ("vtail", "F")}, new)
    st = {k: o[k] for k in o if k not in ("vr", "vsm")}
    vreach = sum(o["vr"])
    vsmall = o["vsm"][0] - o["vsm"][1]

    LM = [sum((st["H"][c] for c in X if LOOPDIR[c] == m), zero(net.t)) for m in range(4)]
    sp_ = [st["score"][m] + (D + 2) * (1 - st["safe"][m]) for m in range(4)]

    def compare(vals, smaller_wins):
        out = {}
        for m in range(4):
            for m2 in range(4):
                if m2 == m:
                    continue
                z = (vals[m2] - vals[m]) if smaller_wins else (vals[m] - vals[m2])
                out[(m, m2)] = [z, z - 1] if m2 < m else [z + 1, z]
        return out
    nsafe = sum(st["safe"])
    nlegal = sum(st["legal"])
    new = {
        "vrb": [vreach, vreach - 1],
        "vsmall": vsmall,
        "far": compare(sp_, True),
        "v3c": compare(st["v3log"], False),
        "anysafe": [nsafe, nsafe - 1],
        "anylegal": [nlegal, nlegal - 1],
        "looplegal": [LM[m] + st["legal"][m] - 1 for m in range(4)],
        "firstlegal": [st["legal"][m] - sum((st["legal"][m2] for m2 in range(m)), zero(net.t)) for m in range(4)],
    }
    keep = {k: st[k] for k in ("H", "pathok", "geK", "ord", "lok", "PF")}
    o = net.step(keep, new)
    st = {k: o[k] for k in keep}
    vreachB = o["vrb"][0] - o["vrb"][1]
    vsm = o["vsmall"]
    far = {key: u[0] - u[1] for key, u in o["far"].items()}
    v3c = {key: u[0] - u[1] for key, u in o["v3c"].items()}
    anysafe = o["anysafe"][0] - o["anysafe"][1]
    anylegal = o["anylegal"][0] - o["anylegal"][1]
    cond = 1 - anylegal + sum(o["looplegal"])
    firstlegal = o["firstlegal"]
    LM = [sum((st["H"][c] for c in X if LOOPDIR[c] == m), zero(net.t)) for m in range(4)]

    new = {
        "far": [sum(far[(m, m2)] for m2 in range(4) if m2 != m) - 2 for m in range(4)],
        "v3oh": [sum(v3c[(m, m2)] for m2 in range(4) if m2 != m) - 2 for m in range(4)],
        "andv": vreachB + vsm - 1,
        "vrB": vreachB,
        "vsm": vsm,
        "anysafe": anysafe,
        "nosafe": [[LM[m] + cond - 1, firstlegal[m] - cond] for m in range(4)],
        "mode1": st["geK"] + st["ord"] - 1,
        "mode2": st["geK"] + (1 - st["ord"]) + st["lok"] - 2,
    }
    keep = {k: st[k] for k in ("H", "pathok", "PF")}
    o = net.step(keep, new)
    st = o
    nosafe = [o["nosafe"][m][0] + o["nosafe"][m][1] for m in range(4)]
    orv = o["vrB"] + o["vsm"] - o["andv"]

    new = {
        "stall": [[o["far"][m] + o["anysafe"] - 1, nosafe[m] - o["anysafe"]] for m in range(4)],
        "free": st["pathok"] + orv - 1,
    }
    keep = {k: st[k] for k in ("H", "PF", "mode1", "mode2", "v3oh")}
    o = net.step(keep, new)
    LM = [sum((o["H"][c] for c in X if LOOPDIR[c] == m), zero(net.t)) for m in range(4)]
    stall = [o["stall"][m][0] + o["stall"][m][1] for m in range(4)]
    rest = 1 - o["mode1"] - o["mode2"]
    new = {"pieces": [[o["mode1"] + o["v3oh"][m] - 1,
                       o["mode2"] + LM[m] - 1,
                       rest + o["free"] + o["PF"][m] - 2,
                       rest - o["free"] + stall[m] - 1] for m in range(4)]}
    o = net.step({}, new)
    net.step({}, {"out": [sum(o["pieces"][m]) for m in range(4)]})
    return net


def to_matrices(net):
    mats = []
    n_in = net.n_in
    for rows, cols, vals, bias in net.layers:
        n_out = len(bias)
        W = sp.csr_matrix((np.array(vals, dtype=np.float64), (rows, cols)), shape=(n_out, n_in))
        mats.append((W, np.array(bias, dtype=np.float64)))
        n_in = n_out
    return mats


def encode(body, food):
    x = np.zeros(3 * CELLS)
    x[body[0]] = 1
    x[CELLS + food] = 1
    L = len(body)
    for i, c in enumerate(body):
        x[2 * CELLS + c] = L - i
    return x


def forward(mats, x):
    for W, b in mats:
        x = np.maximum(W @ x + b, 0)
    return x


if __name__ == "__main__":
    import pickle
    import time
    t0 = time.time()
    net = build()
    mats = to_matrices(net)
    nnz = sum(W.nnz + int((b != 0).sum()) for W, b in mats)
    print(f"layers: {len(mats)}, nonzero weights: {nnz:,}, widest layer: {max(W.shape[0] for W, _ in mats)}, "
          f"built in {time.time() - t0:.0f}s")
    pickle.dump(mats, open("v4_net.pkl", "wb"))
