"""Hand-written Snake network, version 5 (the rule is in v5_rule.py).
No training, no gradients: every weight is set from the design below.

Input (300): head one-hot, food one-hot, age of every cell.
Every unit is ReLU(weights . previous layer + bias); all numbers are whole.

  facts  about the snake and, for each of 64 loops: is the body in order along
         it, does the way ahead clear in time; then the loop move.
  flood  24 layers: path flood from the head (body cells open as they empty)
         plus a tail check for each of the 4 moves, one cell per layer.
  trace  24 layers: walk the shortest path back from the food.
  flood  24 layers: could the snake reach its tail after eating?
  end    choose: in-order loop / clearing loop / path / stall.
"""
import pickle

import v5_rule as R
from build_v4 import A, E, Net, encode, forward, to_matrices, zero  # shared builder
from v5_rule import NB, NL, NXT, POS, TAB

C, K, D, M = 100, R.K, R.DMAX, R.MARGINS[0]
X, J, MV = range(C), range(len(TAB)), range(4)
PAIRS = [(x, m) for x in X for m in MV if NB[x][m] is not None]
FROM = [[None] * 4 for _ in X]
for _x, _m in PAIRS:
    FROM[NB[_x][_m]][_m] = _x
DIR = [[NB[c].index(NXT[j][c]) for c in X] for j in J]


def lin(exprs, coefs, t):
    """Weighted sum of expressions, built in one go."""
    c, k = {}, 0
    for e, w in zip(exprs, coefs):
        if w:
            assert e.t == t
            for i, v in e.c.items():
                c[i] = c.get(i, 0) + v * w
            k += e.k * w
    return E(t, c, k)


def total(exprs, t):
    exprs = list(exprs)
    return lin(exprs, [1] * len(exprs), t)


def b2(u):
    """Two units ReLU(z), ReLU(z-1) make a 0/1 step for whole numbers."""
    return u[0] - u[1]


def eq0(u):
    """Three units ReLU(z+1), ReLU(z), ReLU(z-1) make [z == 0]."""
    return u[0] - 2 * u[1] + u[2]


def compare(vals, smaller):
    out = {}
    for m in MV:
        for q in MV:
            if q != m:
                z = vals[q] - vals[m] if smaller else vals[m] - vals[q]
                out[(m, q)] = [z, z - 1] if q < m else [z + 1, z]
    return out


def build():
    net = Net(3 * C)
    H = [E(0, {x: 1}) for x in X]
    F = [E(0, {C + x: 1}) for x in X]
    a = [E(0, {2 * C + x: 1}) for x in X]

    def go(st, new, drop=()):
        o = net.step({k: v for k, v in st.items() if k not in drop}, new)
        return {k: o[k] for k in st if k not in drop}, o

    def loopdir(st, j, m):
        return lin(st["H"], [int(DIR[j][c] == m) for c in X], net.t)

    # 1: age thresholds; head and food position along each loop
    st, o = go({"H": H, "F": F, "a": a}, {
        "c": [[v - 1, v - 2, v - 3] for v in a],
        "ph": [lin(H, POS[j], 0) for j in J],
        "pf": [lin(F, POS[j], 0) for j in J]})
    occ = [st["a"][x] - o["c"][x][0] for x in X]
    blk = [o["c"][x][0] - o["c"][x][1] for x in X]
    eq2 = [blk[x] - o["c"][x][1] + o["c"][x][2] for x in X]
    st.update(occ=occ, blk=blk, eq2=eq2, T=[occ[x] - blk[x] for x in X], ph=o["ph"], pf=o["pf"])

    # 2: wrap-around indicators, consecutive body cells
    L = total(st["occ"], net.t)
    ph = st["ph"]
    st, o = go(st, {
        "wf": [[ph[j] - st["pf"][j], ph[j] - st["pf"][j] - 1] for j in J],
        "wx": [[[ph[j] - POS[j][x], ph[j] - POS[j][x] - 1] for x in X] for j in J],
        "pt": [lin(st["T"], POS[j], net.t) for j in J],
        "eq": {p: [st["a"][NB[p[0]][p[1]]] - st["a"][p[0]] - i for i in (0, 1, 2)] for p in PAIRS},
        "gk": [L - K + 1, L - K]})
    st["geK"] = b2(o["gk"])
    st["Df"] = [st["pf"][j] - st["ph"][j] + 100 * b2(o["wf"][j]) for j in J]
    pt, eq, wx = o["pt"], {p: eq0(u) for p, u in o["eq"].items()}, o["wx"]

    # 3: per loop, body cells that would not clear in time; legal moves
    def late(j, x, mg):
        d = POS[j][x] - st["ph"][j] + 100 * b2(wx[j][x])
        return st["a"][x] + mg - d + A * (st["occ"][x] - 1) - A * st["H"][x]
    new = {"v3": [[late(j, x, M) for x in X] for j in J],
           "v0": [[late(j, x, 0) for x in X] for j in J],
           "e": {p: eq[p] + st["occ"][p[0]] - 1 for p in PAIRS},
           "wt": [[st["ph"][j] - pt[j], st["ph"][j] - pt[j] - 1] for j in J],
           "pt": pt,
           "legal": [[st["H"][FROM[x][m]] - st["blk"][x] for x in X if FROM[x][m] is not None] for m in MV],
           "eat": [[st["H"][FROM[x][m]] + st["F"][x] - 1 for x in X if FROM[x][m] is not None] for m in MV]}
    st, o = go(st, new, drop=("pf",))
    t = net.t
    st["ah"] = [o["pt"][j] - st["ph"][j] + 100 * b2(o["wt"][j]) for j in J]
    st["legal"] = [total(o["legal"][m], t) for m in MV]
    eat = [total(o["eat"][m], t) for m in MV]
    ks = list(o["e"])
    S = [lin([o["e"][p] for p in ks], [TAB[j][x][NB[x][m]] for x, m in ks], t) for j in J]
    L = total(st["occ"], t)

    # 4: loop verdicts; what each of the 4 moves leaves behind
    new = {"ok3": [1 - total(o["v3"][j], t) for j in J],
           "ok0": [1 - total(o["v0"][j], t) for j in J],
           "ord": [[100 - S[j], 99 - S[j]] for j in J],
           "pu": [[st["eq2"][x] - eat[m] for x in X] for m in MV],
           "tb": [[st["T"][x] + eat[m] - 1 for x in X] for m in MV],
           "sm": [[3 - L - eat[m], 2 - L - eat[m]] for m in MV]}
    st, o = go(st, new, drop=("ph", "eq2", "T"))
    st["small"] = [b2(u) for u in o["sm"]]
    st["Tb"] = [[o["pu"][m][x] + o["tb"][m][x] for x in X] for m in MV]
    st["pm"] = [[1 - st["blk"][x] + o["pu"][m][x] for x in X] for m in MV]

    # 5: the first loop of each kind
    def first(v):
        return [v[j] - total(v[:j], net.t) for j in J]
    st, o = go(st, {"fo": first([b2(u) for u in o["ord"]]), "f3": first(o["ok3"]), "f0": first(o["ok0"])})
    t = net.t
    fo, f3, f0 = o["fo"], o["f3"], o["f0"]
    st["anyO"], st["any3"], st["any0"] = total(fo, t), total(f3, t), total(f0, t)

    # 6: read the chosen loop's numbers (gate by the choice, then add up)
    def pick(sel, vals):
        return [vals[j] + A * (sel[j] - 1) for j in J]

    def jump(j, m):
        return lin(st["H"], [TAB[j][c][NB[c][m]] if NB[c][m] is not None else 0 for c in X], t)
    new = {"x": [pick(fo, [jump(j, m) for j in J]) for m in MV],
           "dfs": pick(fo, st["Df"]), "ahs": pick(fo, st["ah"])}
    for n, sel in (("no", fo), ("n3", f3), ("n0", f0)):
        new[n] = [[loopdir(st, j, m) + sel[j] - 1 for j in J] for m in MV]
    st, o = go(st, new, drop=("Df", "ah"))
    t = net.t
    jp = [total(o["x"][m], t) for m in MV]
    dfs, ahs = total(o["dfs"], t), total(o["ahs"], t)
    no, n3, n0 = ([total(o[n][m], t) for m in MV] for n in ("no", "n3", "n0"))
    L = total(st["occ"], t)

    # 7-10: longest valid shortcut on the chosen loop, else its plain move
    room = [2 * ahs - 2 * jp[m] - 2 * R.BUF - L for m in MV]
    st, o = go(st, {"x": jp, "no": no, "n3": n3, "n0": n0,
                    "g": [[[jp[m] - 1, jp[m] - 2], [dfs - jp[m] + 1, dfs - jp[m]],
                           [room[m], room[m] - 1]] for m in MV]})
    st["n3"], st["n0"] = o["n3"], o["n0"]
    val = [total([b2(u) for u in o["g"][m]], net.t) for m in MV]
    st, o = go(st, {"s": [o["x"][m] + A * (val[m] - 3) for m in MV], "no": o["no"]})
    s = [o["s"][m] + o["no"][m] for m in MV]  # plain move scores 1, shortcuts >= 2
    st, o = go(st, {"cmp": compare(s, False)})
    st, o = go(st, {"oh": [total([b2(o["cmp"][(m, q)]) for q in MV if q != m], net.t) - 2 for m in MV]})
    st["oh"] = o["oh"]

    # floods: timed path flood and 4 tail checks. Cell x is open at step k when
    # its age <= k, kept as the pair ReLU(a-k), ReLU(a-k-1).
    t = net.t
    st, o = go(st, {"op": [[st["a"][x] - 1, st["a"][x] - 2] for x in X], "N": st["H"]})
    t = net.t
    op, N = o["op"], o["N"]
    Rp, acc = [zero(t)] * C, [zero(t)] * C  # reached before this step / count
    Rm = [[st["H"][FROM[x][m]] if FROM[x][m] is not None else zero(t) for x in X] for m in MV]
    am = None
    del st["blk"]
    for k in range(1, D + 1):
        t = net.t
        RN = [Rp[x] + N[x] for x in X]
        P = [1 - b2(u) for u in op]
        sp = [total([N[y] for y in NL[x]], t) + A * (P[x] - 1) - A * RN[x] for x in X]
        new = {"op": [[u[1], u[1] - 1] for u in op], "Rp": RN,
               "acc": [acc[x] + RN[x] for x in X],
               "U": [[sp[x], sp[x] - 1] for x in X],
               "am": [[zero(t)] * C for _ in MV] if am is None
               else [[am[m][x] + Rm[m][x] for x in X] for m in MV],
               "V": []}
        for m in MV:
            sm = [total([Rm[m][x]] + [Rm[m][y] for y in NL[x]], t) + A * (st["pm"][m][x] - 1) for x in X]
            new["V"].append([[sm[x], sm[x] - 1] for x in X])
        st, o = go(st, new)
        Rp, acc, op, am = o["Rp"], o["acc"], o["op"], o["am"]
        N = [b2(u) for u in o["U"]]
        Rm = [[b2(u) for u in o["V"][m]] for m in MV]
    # add the last step; distance = D + 1 - acc (the head itself: 0)
    acc = [acc[x] + Rp[x] + N[x] for x in X]
    am = [[am[m][x] + Rm[m][x] for x in X] for m in MV]

    # path bookkeeping and the verdict on each move
    new = {"eq": {(x, m): [acc[x] - acc[NB[x][m]] + i for i in (2, 1, 0)] for x, m in PAIRS},
           "d2": [[D - acc[x], D - 1 - acc[x]] for x in X],
           "d1": [[acc[x] - D + 1, acc[x] - D, acc[x] - D - 1] for x in X],
           "aF": [acc[x] + A * (st["F"][x] - 1) for x in X],
           "aT": [[am[m][x] + A * (st["Tb"][m][x] - 1) for x in X] for m in MV],
           "rc": [[st["Tb"][m][x] + Rm[m][x] - 1 for x in X] for m in MV]}
    st, o = go(st, new, drop=("pm", "Tb"))
    t = net.t
    eq = {p: eq0(u) for p, u in o["eq"].items()}
    st["d1"] = [eq0(u) for u in o["d1"]]
    st["aF"] = total(o["aF"], t)
    aT = [total(v, t) for v in o["aT"]]
    rc = [total(v, t) for v in o["rc"]]
    st, o = go(st, {"eq": {p: eq[p] + b2(o["d2"][p[0]]) - 1 for p in PAIRS},
                    "rF": [st["aF"], st["aF"] - 1], "rc": rc,
                    "ra": [rc[m] + st["small"][m] - 1 for m in MV],
                    "sc": [aT[m] - A * st["small"][m] for m in MV]})
    st["pathok"] = b2(o["rF"])
    eqm = o["eq"]
    rok = [o["rc"][m] + st["small"][m] - o["ra"][m] for m in MV]
    nx = {(x, m): eqm[(x, m)] - total([eqm[(x, q)] for q in range(m) if (x, q) in eqm], net.t)
          for x, m in PAIRS}
    st, o = go(st, {"nx": nx, "Q": [st["F"][x] + st["pathok"] - 1 for x in X],
                    "safe": [st["legal"][m] + rok[m] - 1 for m in MV],
                    "score": [o["sc"][m] + (D + 1) * st["small"][m] for m in MV]}, drop=("small",))
    st.update(safe=o["safe"], score=o["score"])
    nx, Q = o["nx"], o["Q"]

    # trace: cells on the path count up once per later layer, so JJ ends as
    # D - 1 - (steps back from the food) and every trace layer is the same
    ON, JJ = [zero(net.t)] * C, [zero(net.t)] * C
    for i in range(D):
        st, o = go(st, {"nx": nx, "w": {p: Q[p[0]] + nx[p] - 1 for p in PAIRS},
                        "ON": [ON[x] + Q[x] for x in X], "JJ": [JJ[x] + ON[x] for x in X]})
        nx, ON, JJ = o["nx"], o["ON"], o["JJ"]
        Q = [total([o["w"][(FROM[y][m], m)] for m in MV if FROM[y][m] is not None], net.t) for y in X]

    # the imagined body once the food is eaten, and the path's first step
    L = total(st["occ"], net.t)
    st, o = go(st, {"on": [L + 1 - (D - 1) + JJ[x] + A * (ON[x] - 1) for x in X],
                    "old": [st["a"][x] - D + st["aF"] - A * ON[x] for x in X],
                    "FS": [ON[x] + st["d1"][x] - 1 for x in X]}, drop=("d1", "aF", "a"))
    vb, FS = [o["on"][x] + o["old"][x] for x in X], o["FS"]
    st, o = go(st, {"v": [[vb[x], vb[x] - 1, vb[x] - 2] for x in X],
                    "PF": [[st["H"][c] + FS[NB[c][m]] - 1 for c in X if NB[c][m] is not None] for m in MV]})
    st["PF"] = [total(v, net.t) for v in o["PF"]]
    st["vt"] = [eq0(u) for u in o["v"]]
    st["vp"] = [1 - u[1] + u[2] for u in o["v"]]
    Rv = st["F"]
    for k in range(D):
        sv = [total([Rv[x]] + [Rv[y] for y in NL[x]], net.t) + A * (st["vp"][x] - 1) for x in X]
        st, o = go(st, {"V": [[sv[x], sv[x] - 1] for x in X]})
        Rv = [b2(u) for u in o["V"]]

    # end: choose the mode and the move
    L = total(st["occ"], net.t)
    st, o = go(st, {"vr": [st["vt"][x] + Rv[x] - 1 for x in X], "vs": [2 - L, 1 - L]},
               drop=("vt", "vp", "F", "occ"))
    vr = total(o["vr"], net.t) + b2(o["vs"])
    ns, nl = total(st["safe"], net.t), total(st["legal"], net.t)
    LM = [loopdir(st, 0, m) for m in MV]
    st, o = go(st, {"vr": [vr, vr - 1], "ns": [ns, ns - 1], "nl": [nl, nl - 1],
                    "far": compare([st["score"][m] + (D + 2) * (1 - st["safe"][m]) for m in MV], True),
                    "ll": [LM[m] + st["legal"][m] - 1 for m in MV],
                    "fl": [st["legal"][m] - total(st["legal"][:m], net.t) for m in MV], "LM": LM},
               drop=("H", "score", "safe", "legal"))
    t = net.t
    st["LM"] = o["LM"]
    cond = 1 - b2(o["nl"]) + total(o["ll"], t)  # loop 0's move when nothing is safe
    st["anysafe"], vrB = b2(o["ns"]), b2(o["vr"])
    far = [total([b2(o["far"][(m, q)]) for q in MV if q != m], t) - 2 for m in MV]
    lng = st["geK"] - st["anyO"]
    st, o = go(st, {"far": far, "ns": [[st["LM"][m] + cond - 1, o["fl"][m] - cond] for m in MV],
                    "free": st["pathok"] + vrB - 1,
                    "m1": st["geK"] + st["anyO"] - 1, "m2": lng + st["any3"] - 1,
                    "m3": lng - st["any3"] + st["any0"] - st["anysafe"] - 1},
               drop=("geK", "anyO", "any3", "any0", "pathok", "LM"))
    rest = 1 - o["m1"] - o["m2"] - o["m3"]
    nos = [o["ns"][m][0] + o["ns"][m][1] for m in MV]
    st, o = go(st, {"m1": o["m1"], "m2": o["m2"], "m3": o["m3"], "go": rest + o["free"] - 1,
                    "stay": rest - o["free"],
                    "stall": [[o["far"][m] + st["anysafe"] - 1, nos[m] - st["anysafe"]] for m in MV]},
               drop=("anysafe",))
    o = net.step({}, {"p": [[o["m1"] + st["oh"][m] - 1, o["m2"] + st["n3"][m] - 1,
                             o["m3"] + st["n0"][m] - 1, o["go"] + st["PF"][m] - 1,
                             o["stay"] + total(o["stall"][m], net.t) - 1] for m in MV]})
    net.step({}, {"out": [total(o["p"][m], net.t) for m in MV]})
    return net


if __name__ == "__main__":
    mats = to_matrices(build())
    nnz = sum(W.nnz + int((b != 0).sum()) for W, b in mats)
    print(f"layers {len(mats)}, nonzero weights {nnz:,}, widest {max(W.shape[0] for W, _ in mats)}")
    pickle.dump(mats, open("v5_net.pkl", "wb"))
