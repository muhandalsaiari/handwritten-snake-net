"""Hand-written Snake network, version 7 (the rule is in v7_rule.py).
No training, no gradients: every weight is set from the design below.

Input (300): head one-hot, food one-hot, age of every cell.
Every unit is ReLU(weights . previous layer + bias); all numbers are whole.
Three parts run side by side, then the last layers choose between them:

  core       v6's block driving: path floods, traced candidates, "does the body
             still fit some loop", plus a 10-step flood from each candidate's
             end that measures how close the remaining free cells are
  free_play  early play in all 4 directions with a tail check
  loops      which of 8 loops clears ahead, for folding into block driving
"""
import pickle

import v7_rule as V
from build_v4 import A, E, Net, encode, forward, to_matrices, zero  # shared builder
from build_v5 import b2, compare, eq0, lin, total
from v7_rule import BL, EDGE, MV, NB, NL, PAIRS, POS, PRED

C, D, BIG = 100, V.DMAX, 20000
X, EN, B = range(C), range(40), range(25)
STEPS = [(x, t) for x in X for t in (0, 1) if MV[x][t] is not None]  # directed moves
INTO = [[(y, t) for y, t in PRED[x]] for x in X]
CELLS_OF = [[x for x in X if BL[x] == b] for b in B]
DECIDERS = [[x for x in X if EDGE[x] == e] for e in EN]
M4 = range(4)
PAIR4 = [(x, m) for x in range(100) for m in M4 if NB[x][m] is not None]
FROM4 = [[None] * 4 for _ in range(100)]
for _x, _m in PAIR4:
    FROM4[NB[_x][_m]][_m] = _x
CAND = [("p", 0), ("p", 1), ("c", 0), ("c", 1)]


def step01(z):
    """The pair of units that becomes [z >= 1]."""
    return [z, z - 1]


def core(net, H, F, a):
    def go(st, new, drop=()):
        k, n = yield ({k: v for k, v in st.items() if k not in drop}, new)
        return k, n

    # 1: age thresholds, consecutive body cells, head/food on a move
    st, o = yield from go({"H": H, "F": F, "a": a}, {
        "c": [[v - 1, v - 2] for v in a],
        "eq": {s: [a[MV[s[0]][s[1]]] - a[s[0]] - i for i in (0, 1, 2)] for s in STEPS}})
    occ = [st["a"][x] - o["c"][x][0] for x in X]
    blk = [o["c"][x][0] - o["c"][x][1] for x in X]
    st.update(occ=occ, blk=blk)
    eq = {s: eq0(u) for s, u in o["eq"].items()}
    # 2: the body's own moves; which of IN / OUT is free; does it eat
    st, o = yield from go(st, {
        "bm": {s: eq[s] + st["occ"][s[0]] - 1 for s in STEPS},
        "free": [[st["H"][x] - st["blk"][MV[x][t]] for x in X if MV[x][t] is not None] for t in (0, 1)],
        "eat": [[st["H"][x] + st["F"][MV[x][t]] - 1 for x in X if MV[x][t] is not None] for t in (0, 1)],
        "vb": [step01(total([st["occ"][x] for x in CELLS_OF[b]], net.t)) for b in B]}, drop=())
    t = net.t
    bm = o["bm"]                           # body cell moved IN / OUT to its next cell
    st["bm"] = bm
    st["free"] = [total(v, t) for v in o["free"]]
    st["eat"] = [total(v, t) for v in o["eat"]]
    vis = [b2(u) for u in o["vb"]]
    out = [total([bm[(x, 1)] for x in DECIDERS[e]], t) for e in EN]
    inn = [total([bm[(x, 0)] for x in DECIDERS[e]], t) for e in EN]
    # 3: pairs joined / kept apart by the body now; what is left after one move
    new = {"d1": [step01(out[e]) for e in EN], "d0": [step01(inn[e]) for e in EN], "vis": vis}
    # after a move the old cell stays if age >= 2, or age 1 when eating
    new["stay"] = [[st["occ"][x] - (st["occ"][x] - st["blk"][x]) + 0 for x in X],      # age >= 2
                   [st["occ"][x] - st["blk"][x] for x in X]]                             # the tail
    new["hd"] = [[st["H"][x] for x in X if EDGE[x] == e] for e in EN]
    new["nk"] = [st["occ"][x] - st["H"][x] - total([bm[(x, q)] for q in (0, 1) if (x, q) in bm], net.t) for x in X]
    Lx = total(st["occ"], net.t)
    new["L"], new["gk"] = Lx, [Lx - V.K + 1, Lx - V.K]
    st, o = yield from go(st, new, drop=("blk",))
    st["nk"], st["L"], st["geK"] = o["nk"], o["L"], b2(o["gk"])
    t = net.t
    d1, d0 = [b2(u) for u in o["d1"]], [b2(u) for u in o["d0"]]
    vis = o["vis"]
    old2, tail = o["stay"]
    st["d1"], st["d0"], st["vis"], st["old2"], st["tail"] = d1, d0, vis, old2, tail
    # 4: conforming moves; surviving cells and their decisions for IN / OUT
    allow = {}
    new = {"nkr": step01(total(st["nk"], t)), "tl": [[st["tail"][x] + st["eat"][m] - 1 for x in X] for m in (0, 1)]}
    for (x, m) in STEPS:
        e = EDGE[x]
        if e is None:
            continue
        if m == 0:
            allow[(x, m)] = 1 - st["d1"][e]
        else:
            # not kept apart, and (joined or the next block is unvisited)
            allow[(x, m)] = 1 - st["d0"][e] - st["vis"][BL[MV[x][1]]] + st["d1"][e]
    new["al"] = {s: step01(allow[s]) for s in allow}
    st, o = yield from go(st, new, drop=("d1", "d0", "vis"))
    t = net.t
    st["nkr"] = b2(o["nkr"])
    al = {s: (b2(o["al"][s]) if s in o["al"] else E(t, {}, 1)) for s in STEPS}
    st["al"] = al
    keep = [[st["old2"][x] + o["tl"][m][x] for x in X] for m in (0, 1)]  # still body after move m
    # 5: per move m, the decisions left
    new = {"k": [{s: keep[m][s[0]] + st["bm"][s] - 1 for s in STEPS if EDGE[s[0]] is not None} for m in (0, 1)],
           "kb": [[step01(total([keep[m][x] for x in CELLS_OF[b]], t)
                          + total([st["H"][x] for x in X if MV[x][m] is not None and BL[MV[x][m]] == b]
                                  + [st["H"][x] for x in CELLS_OF[b]], t)) for b in B] for m in (0, 1)]}
    st, o = yield from go(st, new, drop=("old2", "tail"))
    t = net.t
    k, st["mv"] = o["k"], [[b2(u) for u in o["kb"][m]] for m in (0, 1)]
    new = {}
    for nm, want in (("m1", 1), ("m0", 0)):
        new[nm] = [[step01(total([k[m][(x, want)] for x in DECIDERS[e] if (x, want) in k[m]]
                                 + ([st["H"][x] for x in DECIDERS[e]] if m == want else []), t))
                    for e in EN] for m in (0, 1)]
    st, o = yield from go(st, new)
    st["m1"] = [[b2(u) for u in o["m1"][m]] for m in (0, 1)]
    st["m0"] = [[b2(u) for u in o["m0"][m]] for m in (0, 1)]

    # ---- floods. A cell newly reached sends a message along its moves.
    t = net.t
    st, o = yield from go(st, {"op": [[st["a"][x] - 1, st["a"][x] - 2] for x in X], "N": st["H"],
                    "W": {s: st["H"][s[0]] + st["al"][s] - 1 for s in STEPS}})
    t = net.t
    op = o["op"]
    Np, Nc, Wc = o["N"], list(o["N"]), o["W"]
    Rp = {"p": [zero(t)] * C, "c": [zero(t)] * C}
    acc = {"p": [zero(t)] * C, "c": [zero(t)] * C}
    for i in range(1, D + 1):
        t = net.t
        P = [1 - b2(u) for u in op]
        RN = {"p": [Rp["p"][x] + Np[x] for x in X], "c": [Rp["c"][x] + Nc[x] for x in X]}
        sp = [total([Np[y] for y, _ in INTO[x]], t) + A * (P[x] - 1) - A * RN["p"][x] for x in X]
        sc = [total([Wc[s] for s in INTO[x]], t) + A * (P[x] - 1) - A * RN["c"][x] for x in X]
        st, o = yield from go(st, {"op": [[u[1], u[1] - 1] for u in op], "R": RN,
                        "acc": {n: [acc[n][x] + RN[n][x] for x in X] for n in "pc"},
                        "Up": [step01(sp[x]) for x in X], "Uc": [step01(sc[x]) for x in X],
                        "Wc": {s: step01(sc[s[0]] + A * (st["al"][s] - 1)) for s in STEPS}})
        op, Rp, acc = o["op"], o["R"], o["acc"]
        Np, Nc = [b2(u) for u in o["Up"]], [b2(u) for u in o["Uc"]]
        Wc = {s: b2(u) for s, u in o["Wc"].items()}
    acc = {"p": [acc["p"][x] + Rp["p"][x] + Np[x] for x in X],
           "c": [acc["c"][x] + Rp["c"][x] + Nc[x] for x in X]}
    # distance = D + 1 - acc; a move y -> x is on a shortest path if acc[y] = acc[x] + 1
    t = net.t
    st, o = yield from go(st, {"eq": {n: {s: [acc[n][s[0]] - acc[n][MV[s[0]][s[1]]] - i for i in (0, 1, 2)]
                               for s in STEPS} for n in "pc"},
                    "aF": {n: [acc[n][x] + A * (st["F"][x] - 1) for x in X] for n in "pc"}})
    t = net.t
    e_ = {n: {s: eq0(u) for s, u in o["eq"][n].items()} for n in "pc"}
    aF = {n: total(o["aF"][n], t) for n in "pc"}
    st, o = yield from go(st, {"e": {"p": e_["p"], "c": {s: e_["c"][s] + st["al"][s] - 1 for s in STEPS}},
                    "aF": aF, "rF": {n: step01(aF[n]) for n in "pc"}}, drop=("al",))
    st["aF"] = o["aF"]
    e_ = o["e"]
    st["rF"] = {n: b2(o["rF"][n]) for n in "pc"}
    nx = {}
    for j, (n, last) in enumerate(CAND):
        nx[j] = {}
        for x in X:
            pr = INTO[x][::-1] if last else INTO[x]
            for i, s in enumerate(pr):
                nx[j][s] = e_[n][s] - total([e_[n][q] for q in pr[:i]], net.t)
    st, o = yield from go(st, {"nx": nx, "Q": {j: [st["F"][x] + st["rF"][n] - 1 for x in X]
                                    for j, (n, _) in enumerate(CAND)}})
    nx, Q = o["nx"], o["Q"]

    # ---- trace: ON = move used by the path, JJ counts layers since then
    t = net.t
    ON = {j: {s: zero(t) for s in STEPS} for j in range(4)}
    JJ = {j: {s: zero(t) for s in STEPS} for j in range(4)}
    for i in range(D):
        w = {j: {s: Q[j][MV[s[0]][s[1]]] + nx[j][s] - 1 for s in STEPS} for j in range(4)}
        st, o = yield from go(st, {"nx": nx, "w": w,
                        "ON": ON, "JJ": {j: {s: JJ[j][s] + ON[j][s] for s in STEPS} for j in range(4)}})
        nx, JJ = o["nx"], o["JJ"]
        # count the new moves in at once (they are carried as one unit next layer)
        ON = {j: {s: o["ON"][j][s] + o["w"][j][s] for s in STEPS} for j in range(4)}
        Q = {j: [total([o["w"][j][(x, m)] for m in (0, 1) if (x, m) in o["w"][j]], net.t) for x in X]
             for j in range(4)}
    # a move found at step i (0 = into the food) has JJ = D - 1 - i; its cell
    # has age L - i once the food is eaten, so it is still body if that is >= 1
    L = total(st["occ"], net.t)
    new = {"ae": {j: {s: step01(L - (D - 1) + JJ[j][s] + A * (ON[j][s] - 1)) for s in STEPS} for j in range(4)},
           # an old cell is still body if age - k + 1 >= 1, k = D + 1 - aF
           "ao": {n: [step01(st["a"][x] - D + st["aF"][n] + A * (st["rF"][n] - 1)) for x in X] for n in "pc"},
           "vp": {j: {s: L - (D - 1) + JJ[j][s] + A * (ON[j][s] - 1) for s in STEPS} for j in range(4)},
           "vo": {n: [st["a"][x] - D + st["aF"][n] + A * (st["rF"][n] - 1) for x in X] for n in "pc"},
           "fo": [[st["H"][x] + ON[j][(x, 1)] - 1 for x in X if (x, 1) in ON[j]] for j in range(4)]}
    st, o = yield from go(st, new, drop=("a", "occ"))
    t = net.t
    ae = {j: {s: b2(u) for s, u in o["ae"][j].items()} for j in range(4)}
    ao = {n: [b2(u) for u in o["ao"][n]] for n in "pc"}
    vage = {j: [total([o["vp"][j][(x, q)] for q in (0, 1) if (x, q) in o["vp"][j]], t)
                + o["vo"][CAND[j][0]][x] for x in X] for j in range(4)}
    st["fo"] = [total(v, t) for v in o["fo"]]   # candidate j starts with OUT
    # old cells keep their decision if they are still body
    new = {"k": {n: {s: ao[n][s[0]] + st["bm"][s] - 1 for s in STEPS if EDGE[s[0]] is not None} for n in "pc"},
           "ae": ae,
           "nb": {n: [ao[n][x] + st["nk"][x] - 1 for x in X] for n in "pc"},
           "op": {j: [[z - 1, z - 2] for z in vage[j]] for j in range(4)},
           "fz": {j: [1 - vage[j][x] - st["F"][x] for x in X] for j in range(4)}, "N": st["F"],
           "vis": {j: [step01(total([ao[CAND[j][0]][x] for x in CELLS_OF[b]]
                                    + [ae[j][s] for s in STEPS if BL[s[0]] == b]
                                    + [st["F"][x] for x in CELLS_OF[b]], t)) for b in B] for j in range(4)}}
    st, o = yield from go(st, new, drop=("bm", "F", "nk"))
    t = net.t
    k, ae = o["k"], o["ae"]
    st["nb"] = {n: total(o["nb"][n], t) for n in "pc"}
    st["op"], st["fz"], st["N"] = o["op"], o["fz"], o["N"]
    visj = {j: [b2(u) for u in o["vis"][j]] for j in range(4)}
    new = {"vis": visj}
    for nm, want in (("j1", 1), ("j0", 0)):
        new[nm] = {j: [step01(total([k[CAND[j][0]][(x, want)] for x in DECIDERS[e]]
                                    + [ae[j][(x, want)] for x in DECIDERS[e]], t)) for e in EN]
                   for j in range(4)}
    st, o = yield from go(st, new)
    J1 = [[b2(u) for u in o["j1"][j]] for j in range(4)] + st["m1"]
    J0 = [[b2(u) for u in o["j0"][j]] for j in range(4)] + st["m0"]
    VS = [o["vis"][j] for j in range(4)] + st["mv"]
    valid = [st["rF"][n] for n, _ in CAND] + st["free"]
    # bad counts: conflicts; joined pairs must equal visited blocks - 1
    t = net.t
    new = {"cf": [[J1[j][e] + J0[j][e] - 1 for e in EN] for j in range(6)],
           "tr": [[total(J1[j], t) - total(VS[j], t) + 1, total(VS[j], t) - 1 - total(J1[j], t)] for j in range(6)],
           "nb": st["nb"], "open": [[1 - J0[j][e] for e in EN] for j in range(6)], "valid": valid}
    st, o = yield from go(st, new, drop=("rF", "m1", "m0", "mv", "eat", "nb"))
    t = net.t
    st["open"] = o["open"]
    st["bad"] = [total(o["cf"][j], t) + total(o["tr"][j], t) + 1 - o["valid"][j]
                 + (o["nb"][CAND[j][0]] if j < 4 else 0) for j in range(6)]

    # ---- connect: flood over blocks from block 0 along the open pairs
    ARC = [(u, w, e) for e, (u, w) in enumerate(PAIRS)] + [(w, u, e) for e, (u, w) in enumerate(PAIRS)]
    INB = [[i for i, (u, w, e) in enumerate(ARC) if w == b] for b in B]
    S = [[E(t, {}, 1 if b == 0 else 0) for b in B] for _ in range(6)]
    op, fz = st.pop("op"), st.pop("fz")
    N = {j: st["N"] for j in range(4)}
    Rp = {j: [zero(t)] * C for j in range(4)}
    acc = {j: [zero(t)] * C for j in range(4)}
    del st["N"]
    for i in range(24):
        new = {}
        if i <= V.FUT:
            RN = {j: [Rp[j][x] + N[j][x] for x in X] for j in range(4)}
            new = {"fz": fz, "R": RN, "acc": {j: [acc[j][x] + RN[j][x] for x in X] for j in range(4)},
                   "op": {j: [[u[1], u[1] - 1] for u in op[j]] for j in range(4)},
                   "U": {j: [step01(total([N[j][y] for y, _ in INTO[x]], net.t) - A * b2(op[j][x])
                                    - A * RN[j][x]) for x in X] for j in range(4)}}
        elif i == V.FUT + 1:
            new = {"fs": {j: [V.FUT + 1 - acc[j][x] + A * (fz[j][x] - 1) for x in X] for j in range(4)}}
        m = [[step01(S[j][u] + A * (st["open"][j][e] - 1)) for u, w, e in ARC] for j in range(6)]
        st, o = yield from go(st, {"m": m, **new})
        t = net.t
        if i <= V.FUT:
            fz, Rp, acc, op = o["fz"], o["R"], o["acc"], o["op"]
            N = {j: [b2(u) for u in o["U"][j]] for j in range(4)}
        elif i == V.FUT + 1:
            st["fs"] = [total(o["fs"][j], t) for j in range(4)]
        msg = [[b2(u) for u in o["m"][j]] for j in range(6)]
        S = [[total([msg[j][i] for i in INB[b]], t) + (1 if b == 0 else 0) for b in B] for j in range(6)]
    st, o = yield from go(st, {"un": [[1 - S[j][b] for b in B] for j in range(6)]}, drop=("open",))
    t = net.t
    bad = [st["bad"][j] + total(o["un"][j], t) for j in range(6)]
    de = st["aF"]["p"] - st["aF"]["c"]      # how much longer the conforming path is
    st, o = yield from go(st, {"ok": [1 - bad[j] for j in range(6)], "de": [step01(de - d) for d in range(D)],
                               "w": 98 - st["L"]}, drop=("bad", "aF", "L"))
    ok, fo, fs, w = o["ok"], st["fo"], st["fs"], 2 + 2 * o["w"]
    st, o = yield from go(st, {"g": ok[:2], "s": ok[4:], "c1": ok[1] - ok[0],
                               "gc": [[ok[j] + ok[5] + fo[j] - 2, ok[j] + ok[4] - fo[j] - 1] for j in (2, 3)],
                               "pen": [w + A * (b2(u) - 1) for u in o["de"]]})
    t = net.t
    g = [o["g"][0], o["g"][1]] + [o["gc"][i][0] + o["gc"][i][1] for i in (0, 1)]
    pen = total(o["pen"], t)
    sc = [3 * st["fs"][j] + (pen if j > 1 else 0) + BIG * (1 - g[j]) for j in range(4)]
    sI, sO = o["s"]
    st, o = yield from go(st, {"cmp": compare(sc, True), "any": step01(total(g, t)),
                               "cp": [o["g"][0] + st["fo"][0] - 1, o["c1"] + st["fo"][1] - 1],
                               "anyc": o["g"][0] + o["c1"], "sO": sO,
                               "last": st["free"][1] - sO - sI}, drop=("free", "fs"))
    t = net.t
    any_ = b2(o["any"])
    pk = [total([b2(o["cmp"][(m, q)]) for q in M4 if q != m], t) - 2 for m in M4]
    st, o = yield from go(st, {"pk": [pk[j] + any_ - 1 for j in M4], "wait": [o["sO"] - any_, o["last"] - any_],
                               "OUTc": o["cp"][0] + o["cp"][1], "anyc": o["anyc"]})
    st, o = yield from go(st, {"u": [o["pk"][j] + st["fo"][j] - 1 for j in M4] + o["wait"],
                               "OUTc": o["OUTc"], "anyc": o["anyc"]}, drop=("fo",))
    return {"H": st["H"], "geK": st["geK"], "nkr": st["nkr"], "OUT": total(o["u"], net.t),
            "OUTc": o["OUTc"], "anyc": o["anyc"]}

def free_play(net, H, F, a):
    """Early play in all 4 directions: timed path flood with two tie-breaks,
    'can the tail still be reached', and the stall move far from the tail."""
    def go(st, new, drop=()):
        k, n = yield ({k: v for k, v in st.items() if k not in drop}, new)
        return k, n

    st, o = yield from go({"H": H, "F": F, "a": a}, {"c": [[v - 1, v - 2, v - 3] for v in a]})
    occ = [st["a"][x] - o["c"][x][0] for x in X]
    blk = [o["c"][x][0] - o["c"][x][1] for x in X]
    eq2 = [blk[x] - o["c"][x][1] + o["c"][x][2] for x in X]
    st.update(occ=occ, blk=blk, eq2=eq2, T=[occ[x] - blk[x] for x in X])
    st, o = yield from go(st, {
        "lg": [[st["H"][FROM4[x][m]] - st["blk"][x] for x in X if FROM4[x][m] is not None] for m in M4],
        "et": [[st["H"][FROM4[x][m]] + st["F"][x] - 1 for x in X if FROM4[x][m] is not None] for m in M4]})
    t = net.t
    st["legal"] = [total(v, t) for v in o["lg"]]
    eat = [total(v, t) for v in o["et"]]
    L = total(st["occ"], t)
    st, o = yield from go(st, {"pu": [[st["eq2"][x] - eat[m] for x in X] for m in M4],
                               "tb": [[st["T"][x] + eat[m] - 1 for x in X] for m in M4],
                               "sm": [[3 - L - eat[m], 2 - L - eat[m]] for m in M4],
                               "op": [[st["a"][x] - 1, st["a"][x] - 2] for x in X]}, drop=("eq2", "T"))
    t = net.t
    st["small"] = [b2(u) for u in o["sm"]]
    st["Tb"] = [[o["pu"][m][x] + o["tb"][m][x] for x in X] for m in M4]
    pm = [[1 - st["blk"][x] + o["pu"][m][x] for x in X] for m in M4]
    st["pm"] = pm
    op, N = o["op"], st["H"]
    del st["blk"]
    Rp, acc = [zero(t)] * C, [zero(t)] * C
    Rm = [[st["H"][FROM4[x][m]] if FROM4[x][m] is not None else zero(t) for x in X] for m in M4]
    am = [[zero(t)] * C for _ in M4]
    for i in range(D):
        t = net.t
        RN = [Rp[x] + N[x] for x in X]
        P = [1 - b2(u) for u in op]
        sp = [total([N[y] for y in NL[x]], t) + A * (P[x] - 1) - A * RN[x] for x in X]
        new = {"op": [[u[1], u[1] - 1] for u in op], "R": RN, "acc": [acc[x] + RN[x] for x in X],
               "U": [step01(z) for z in sp],
               "am": am if i == 0 else [[am[m][x] + Rm[m][x] for x in X] for m in M4],
               "V": [[step01(total([Rm[m][x]] + [Rm[m][y] for y in NL[x]], t) + A * (st["pm"][m][x] - 1))
                      for x in X] for m in M4]}
        st, o = yield from go(st, new)
        op, Rp, acc, am = o["op"], o["R"], o["acc"], o["am"]
        N = [b2(u) for u in o["U"]]
        Rm = [[b2(u) for u in o["V"][m]] for m in M4]
    acc = [acc[x] + Rp[x] + N[x] for x in X]
    am = [[am[m][x] + Rm[m][x] for x in X] for m in M4]
    st, o = yield from go(st, {
        "eq": {(x, m): [acc[x] - acc[NB[x][m]] + i for i in (2, 1, 0)] for x, m in PAIR4},
        "d2": [[D - acc[x], D - 1 - acc[x]] for x in X],
        "d1": [[acc[x] - D + 1, acc[x] - D, acc[x] - D - 1] for x in X],
        "aF": [acc[x] + A * (st["F"][x] - 1) for x in X],
        "aT": [[am[m][x] + A * (st["Tb"][m][x] - 1) for x in X] for m in M4],
        "rc": [[st["Tb"][m][x] + Rm[m][x] - 1 for x in X] for m in M4]}, drop=("pm", "Tb"))
    t = net.t
    eq = {p: eq0(u) for p, u in o["eq"].items()}
    st["d1"] = [eq0(u) for u in o["d1"]]
    st["aF"] = total(o["aF"], t)
    aT, rc = [total(v, t) for v in o["aT"]], [total(v, t) for v in o["rc"]]
    st, o = yield from go(st, {"eq": {p: eq[p] + b2(o["d2"][p[0]]) - 1 for p in PAIR4},
                               "rF": step01(st["aF"]), "rc": rc,
                               "ra": [rc[m] + st["small"][m] - 1 for m in M4],
                               "sc": [aT[m] - A * st["small"][m] for m in M4]})
    st["pathok"] = b2(o["rF"])
    e = o["eq"]
    rok = [o["rc"][m] + st["small"][m] - o["ra"][m] for m in M4]
    nx = {0: {}, 1: {}}
    for x, m in PAIR4:
        for j, others in ((0, range(m)), (1, range(m + 1, 4))):
            nx[j][(x, m)] = e[(x, m)] - total([e[(x, q)] for q in others if (x, q) in e], net.t)
    st, o = yield from go(st, {"nx": nx, "Q": {j: [st["F"][x] + st["pathok"] - 1 for x in X] for j in (0, 1)},
                               "safe": [st["legal"][m] + rok[m] - 1 for m in M4],
                               "score": [o["sc"][m] + (D + 1) * st["small"][m] for m in M4]}, drop=("small",))
    st.update(safe=o["safe"], score=o["score"])
    nx, Q = o["nx"], o["Q"]
    t = net.t
    ON = {j: [zero(t)] * C for j in (0, 1)}
    JJ = {j: [zero(t)] * C for j in (0, 1)}
    for i in range(D):
        st, o = yield from go(st, {"nx": nx, "w": {j: {p: Q[j][p[0]] + nx[j][p] - 1 for p in PAIR4} for j in (0, 1)},
                                   "ON": {j: [ON[j][x] + Q[j][x] for x in X] for j in (0, 1)},
                                   "JJ": {j: [JJ[j][x] + ON[j][x] for x in X] for j in (0, 1)}})
        nx, ON, JJ = o["nx"], o["ON"], o["JJ"]
        Q = {j: [total([o["w"][j][(FROM4[y][m], m)] for m in M4 if FROM4[y][m] is not None], net.t) for y in X]
             for j in (0, 1)}
    L = total(st["occ"], net.t)
    st, o = yield from go(st, {"on": {j: [L + 1 - (D - 1) + JJ[j][x] + A * (ON[j][x] - 1) for x in X] for j in (0, 1)},
                               "old": {j: [st["a"][x] - D + st["aF"] - A * ON[j][x] for x in X] for j in (0, 1)},
                               "FS": {j: [ON[j][x] + st["d1"][x] - 1 for x in X] for j in (0, 1)}},
                          drop=("d1", "aF", "a"))
    vb = {j: [o["on"][j][x] + o["old"][j][x] for x in X] for j in (0, 1)}
    FS = o["FS"]
    st, o = yield from go(st, {"v": {j: [[z, z - 1, z - 2] for z in vb[j]] for j in (0, 1)},
                               "PF": {j: [[st["H"][c] + FS[j][NB[c][m]] - 1 for c in X if NB[c][m] is not None]
                                          for m in M4] for j in (0, 1)}})
    st["PF"] = {j: [total(v, net.t) for v in o["PF"][j]] for j in (0, 1)}
    st["vt"] = {j: [eq0(u) for u in o["v"][j]] for j in (0, 1)}
    st["vp"] = {j: [1 - u[1] + u[2] for u in o["v"][j]] for j in (0, 1)}
    Rv = {j: st["F"] for j in (0, 1)}
    for i in range(D):
        st, o = yield from go(st, {"V": {j: [step01(total([Rv[j][x]] + [Rv[j][y] for y in NL[x]], net.t)
                                                    + A * (st["vp"][j][x] - 1)) for x in X] for j in (0, 1)}})
        Rv = {j: [b2(u) for u in o["V"][j]] for j in (0, 1)}
    L = total(st["occ"], net.t)
    st, o = yield from go(st, {"vr": {j: [st["vt"][j][x] + Rv[j][x] - 1 for x in X] for j in (0, 1)},
                               "vs": [2 - L, 1 - L]}, drop=("vt", "vp", "F", "occ"))
    vr = {j: total(o["vr"][j], net.t) + b2(o["vs"]) for j in (0, 1)}
    ns = total(st["safe"], net.t)
    first = [total([st["H"][c] for c in X if next(q for q in M4 if NB[c][q] is not None) == m], net.t) for m in M4]
    nl = total(st["legal"], net.t)
    st, o = yield from go(st, {"vr": {j: step01(vr[j]) for j in (0, 1)}, "ns": step01(ns),
                               "far": compare([st["score"][m] + (D + 2) * (1 - st["safe"][m]) for m in M4], True),
                               "fl": [st["legal"][m] - total(st["legal"][:m], net.t) for m in M4],
                               "fn": [first[m] - nl for m in M4]}, drop=("H", "score", "safe", "legal"))
    t = net.t
    ok = {j: b2(o["vr"][j]) for j in (0, 1)}
    anys = b2(o["ns"])
    far = [total([b2(o["far"][(m, q)]) for q in M4 if q != m], t) - 2 for m in M4]
    rest = [o["fl"][m] + o["fn"][m] for m in M4]
    st, o = yield from go(st, {"g": [st["pathok"] + ok[0] - 1, st["pathok"] + ok[1] - ok[0] - 1], "anys": anys,
                               "far": far, "rest": rest}, drop=("pathok",))
    g = o["g"]
    st, o = yield from go(st, {"d": [[st["PF"][0][m] + g[0] - 1, st["PF"][1][m] + g[1] - 1,
                                      o["far"][m] + o["anys"] - g[0] - g[1] - 1, o["rest"][m] - o["anys"] - g[0] - g[1]]
                                     for m in M4], "anys": o["anys"]}, drop=("PF",))
    return {"dir": [total(v, net.t) for v in o["d"]], "anys": o["anys"]}


def loops(net, H, F, a):
    """Which of the 8 loops clears ahead (with MARGIN to spare, or at all)."""
    def go(st, new, drop=()):
        k, n = yield ({k: v for k, v in st.items() if k not in drop}, new)
        return k, n

    JL = range(len(POS))
    st, o = yield from go({"H": H, "a": a}, {"c": [v - 1 for v in a], "ph": [lin(H, POS[j], 0) for j in JL]})
    st["occ"], ph = [st["a"][x] - o["c"][x] for x in X], o["ph"]
    st, o = yield from go(st, {"ph": ph, "w": [[step01(ph[j] - POS[j][x]) for x in X] for j in JL]})
    late = {mg: [[st["a"][x] + mg - (POS[j][x] - o["ph"][j] + 100 * b2(o["w"][j][x])) + A * (st["occ"][x] - 1)
                  - A * st["H"][x] for x in X] for j in JL] for mg in (V.MARGIN, 0)}
    st, o = yield from go(st, {"l4": late[V.MARGIN], "l0": late[0]}, drop=("a", "occ"))
    t = net.t
    st, o = yield from go(st, {n: [1 - total(o[q][j], t) for j in JL] for n, q in (("k4", "l4"), ("k0", "l0"))})
    t = net.t
    fs = {n: [o[n][j] - total(o[n][:j], t) for j in JL] for n in ("k4", "k0")}
    st, o = yield from go(st, fs)
    t = net.t
    new = {n: [[lin(st["H"], [int(NB[c].index(V.NXT[j][c]) == m) for c in X], t) + o[n][j] - 1 for j in JL]
               for m in M4] for n in ("k4", "k0")}
    any_ = {n + "a": total(o[n], t) for n in ("k4", "k0")}
    st, o = yield from go(st, {**new, **any_}, drop=("H",))
    t = net.t
    return {"n4": [total(v, t) for v in o["k4"]], "n0": [total(v, t) for v in o["k0"]],
            "any4": o["k4a"], "any0": o["k0a"]}


def run(net, gens):
    """Run several parts side by side, layer by layer. A finished part's
    results are simply carried along."""
    n, req, res = len(gens), [None] * len(gens), [None] * len(gens)
    for i, g in enumerate(gens):
        req[i] = next(g)
    while any(r is None for r in res):
        keep = {i: (res[i] if res[i] is not None else req[i]) for i in range(n)}
        o = net.step(keep, {})
        for i, g in enumerate(gens):
            if res[i] is not None:
                res[i] = o[i]
                continue
            try:
                req[i] = g.send(tuple(o[i]))
            except StopIteration as stop:
                res[i] = stop.value
    return res


def build():
    net = Net(3 * C)
    H = [E(0, {x: 1}) for x in X]
    F = [E(0, {C + x: 1}) for x in X]
    a = [E(0, {2 * C + x: 1}) for x in X]
    core_, free_, loop_ = run(net, [core(net, H, F, a), free_play(net, H, F, a), loops(net, H, F, a)])
    c, f, l = core_, free_, loop_
    o = net.step({"c": c, "f": f, "l": l}, {
        "m0": c["geK"] - c["nkr"], "c1": c["geK"] + c["nkr"] + c["anyc"] - 2,
        "c2": c["geK"] + c["nkr"] - c["anyc"] + l["any4"] - 2,
        "c3": c["geK"] + c["nkr"] - c["anyc"] - l["any4"] - f["anys"] + l["any0"] - 2})
    c, f, l = o["c"], o["f"], o["l"]
    o = net.step({"H": c["H"], "f": f["dir"], "n4": l["n4"], "n0": l["n0"], "c2": o["c2"], "c3": o["c3"]}, {
        "o": [o["m0"] + c["OUT"] - 1, o["c1"] + c["OUTc"] - 1],
        "i": [o["m0"] - c["OUT"], o["c1"] - c["OUTc"]],
        "fr": 1 - o["m0"] - o["c1"] - o["c2"] - o["c3"]})
    out, in_ = o["o"][0] + o["o"][1], o["i"][0] + o["i"][1]
    t = net.t

    def heads(m, d):
        return total([o["H"][x] for x in X if MV[x][m] is not None and NB[x][d] == MV[x][m]], t)
    z = [[heads(1, d) + out - 1, heads(0, d) + in_ - 1, o["n4"][d] + o["c2"] - 1, o["n0"][d] + o["c3"] - 1,
          o["f"][d] + o["fr"] - 1] for d in M4]
    o = net.step({}, {"z": z})
    net.step({}, {"out": [total(o["z"][d], net.t) for d in M4]})
    return net


if __name__ == "__main__":
    mats = to_matrices(build())
    nnz = sum(W.nnz + int((b != 0).sum()) for W, b in mats)
    print(f"layers {len(mats)}, nonzero weights {nnz:,}, widest {max(W.shape[0] for W, _ in mats)}")
    pickle.dump(mats, open("v7_net.pkl", "wb"))
