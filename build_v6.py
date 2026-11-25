"""Hand-written Snake network, version 6 (the rule is in v6_rule.py).
No training, no gradients: every weight is set from the design below.

Input (300): head one-hot, food one-hot, age of every cell.
Every unit is ReLU(weights . previous layer + bias); all numbers are whole.

  facts    which pairs of blocks the body joins or keeps apart, legal moves
  flood    20 layers: plain and conforming path floods, one cell per layer
  trace    20 layers: four candidate paths, walked back from the food
  imagine  the body after each candidate, and after a plain IN or OUT move:
           joined pairs, kept-apart pairs, visited blocks
  connect  24 layers: do the remaining open pairs connect all 25 blocks?
  end      first acceptable candidate, else OUT, else IN
"""
import pickle

import v6_rule as R
from build_v4 import A, E, Net, encode, forward, to_matrices, zero  # shared builder
from build_v5 import b2, eq0, lin, total
from v6_rule import BL, EDGE, MV, PAIRS, PRED

C, D = 100, R.DMAX
X, EN, B = range(C), range(40), range(25)
STEPS = [(x, t) for x in X for t in (0, 1) if MV[x][t] is not None]  # directed moves
INTO = [[(y, t) for y, t in PRED[x]] for x in X]
CELLS_OF = [[x for x in X if BL[x] == b] for b in B]
DECIDERS = [[x for x in X if EDGE[x] == e] for e in EN]
CAND = [("p", 0), ("p", 1), ("c", 0), ("c", 1)]


def step01(z):
    """The pair of units that becomes [z >= 1]."""
    return [z, z - 1]


def build():
    net = Net(3 * C)

    def go(st, new, drop=()):
        o = net.step({k: v for k, v in st.items() if k not in drop}, new)
        return {k: o[k] for k in st if k not in drop}, o

    H = [E(0, {x: 1}) for x in X]
    F = [E(0, {C + x: 1}) for x in X]
    a = [E(0, {2 * C + x: 1}) for x in X]

    # 1: age thresholds, consecutive body cells, head/food on a move
    st, o = go({"H": H, "F": F, "a": a}, {
        "c": [[v - 1, v - 2] for v in a],
        "eq": {s: [a[MV[s[0]][s[1]]] - a[s[0]] - i for i in (0, 1, 2)] for s in STEPS}})
    occ = [st["a"][x] - o["c"][x][0] for x in X]
    blk = [o["c"][x][0] - o["c"][x][1] for x in X]
    st.update(occ=occ, blk=blk)
    eq = {s: eq0(u) for s, u in o["eq"].items()}
    # 2: the body's own moves; which of IN / OUT is free; does it eat
    st, o = go(st, {
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
    st, o = go(st, new, drop=("blk",))
    t = net.t
    d1, d0 = [b2(u) for u in o["d1"]], [b2(u) for u in o["d0"]]
    vis = o["vis"]
    old2, tail = o["stay"]
    st["d1"], st["d0"], st["vis"], st["old2"], st["tail"] = d1, d0, vis, old2, tail
    st["occ"] = st["occ"]
    # 4: conforming moves; surviving cells and their decisions for IN / OUT
    allow = {}
    new = {"tl": [[st["tail"][x] + st["eat"][m] - 1 for x in X] for m in (0, 1)]}
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
    st, o = go(st, new, drop=("d1", "d0", "vis"))
    t = net.t
    al = {s: (b2(o["al"][s]) if s in o["al"] else E(t, {}, 1)) for s in STEPS}
    st["al"] = al
    keep = [[st["old2"][x] + o["tl"][m][x] for x in X] for m in (0, 1)]  # still body after move m
    # 5: per move m, the decisions left
    new = {"k": [{s: keep[m][s[0]] + st["bm"][s] - 1 for s in STEPS if EDGE[s[0]] is not None} for m in (0, 1)],
           "kb": [[step01(total([keep[m][x] for x in CELLS_OF[b]], t)
                          + total([st["H"][x] for x in X if MV[x][m] is not None and BL[MV[x][m]] == b]
                                  + [st["H"][x] for x in CELLS_OF[b]], t)) for b in B] for m in (0, 1)]}
    st, o = go(st, new, drop=("old2", "tail"))
    t = net.t
    k, st["mv"] = o["k"], [[b2(u) for u in o["kb"][m]] for m in (0, 1)]
    new = {}
    for nm, want in (("m1", 1), ("m0", 0)):
        new[nm] = [[step01(total([k[m][(x, want)] for x in DECIDERS[e] if (x, want) in k[m]]
                                 + ([st["H"][x] for x in DECIDERS[e]] if m == want else []), t))
                    for e in EN] for m in (0, 1)]
    st, o = go(st, new)
    st["m1"] = [[b2(u) for u in o["m1"][m]] for m in (0, 1)]
    st["m0"] = [[b2(u) for u in o["m0"][m]] for m in (0, 1)]

    # ---- floods. A cell newly reached sends a message along its moves.
    t = net.t
    st, o = go(st, {"op": [[st["a"][x] - 1, st["a"][x] - 2] for x in X], "N": st["H"],
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
        st, o = go(st, {"op": [[u[1], u[1] - 1] for u in op], "R": RN,
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
    st, o = go(st, {"eq": {n: {s: [acc[n][s[0]] - acc[n][MV[s[0]][s[1]]] - i for i in (0, 1, 2)]
                               for s in STEPS} for n in "pc"},
                    "aF": {n: [acc[n][x] + A * (st["F"][x] - 1) for x in X] for n in "pc"}})
    t = net.t
    e_ = {n: {s: eq0(u) for s, u in o["eq"][n].items()} for n in "pc"}
    aF = {n: total(o["aF"][n], t) for n in "pc"}
    st, o = go(st, {"e": {"p": e_["p"], "c": {s: e_["c"][s] + st["al"][s] - 1 for s in STEPS}},
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
    st, o = go(st, {"nx": nx, "Q": {j: [st["F"][x] + st["rF"][n] - 1 for x in X]
                                    for j, (n, _) in enumerate(CAND)}})
    nx, Q = o["nx"], o["Q"]

    # ---- trace: ON = move used by the path, JJ counts layers since then
    t = net.t
    ON = {j: {s: zero(t) for s in STEPS} for j in range(4)}
    JJ = {j: {s: zero(t) for s in STEPS} for j in range(4)}
    for i in range(D):
        w = {j: {s: Q[j][MV[s[0]][s[1]]] + nx[j][s] - 1 for s in STEPS} for j in range(4)}
        st, o = go(st, {"nx": nx, "w": w,
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
           "fo": [[st["H"][x] + ON[j][(x, 1)] - 1 for x in X if (x, 1) in ON[j]] for j in range(4)]}
    st, o = go(st, new, drop=("a", "occ", "aF"))
    t = net.t
    ae = {j: {s: b2(u) for s, u in o["ae"][j].items()} for j in range(4)}
    ao = {n: [b2(u) for u in o["ao"][n]] for n in "pc"}
    st["fo"] = [total(v, t) for v in o["fo"]]   # candidate j starts with OUT
    # old cells keep their decision if they are still body
    new = {"k": {n: {s: ao[n][s[0]] + st["bm"][s] - 1 for s in STEPS if EDGE[s[0]] is not None} for n in "pc"},
           "ae": ae,
           "vis": {j: [step01(total([ao[CAND[j][0]][x] for x in CELLS_OF[b]]
                                    + [ae[j][s] for s in STEPS if BL[s[0]] == b]
                                    + [st["F"][x] for x in CELLS_OF[b]], t)) for b in B] for j in range(4)}}
    st, o = go(st, new, drop=("bm", "F"))
    t = net.t
    k, ae = o["k"], o["ae"]
    visj = {j: [b2(u) for u in o["vis"][j]] for j in range(4)}
    new = {"vis": visj}
    for nm, want in (("j1", 1), ("j0", 0)):
        new[nm] = {j: [step01(total([k[CAND[j][0]][(x, want)] for x in DECIDERS[e]]
                                    + [ae[j][(x, want)] for x in DECIDERS[e]], t)) for e in EN]
                   for j in range(4)}
    st, o = go(st, new)
    J1 = [[b2(u) for u in o["j1"][j]] for j in range(4)] + st["m1"]
    J0 = [[b2(u) for u in o["j0"][j]] for j in range(4)] + st["m0"]
    VS = [o["vis"][j] for j in range(4)] + st["mv"]
    valid = [st["rF"][n] for n, _ in CAND] + st["free"]
    # bad counts: conflicts; joined pairs must equal visited blocks - 1
    t = net.t
    new = {"cf": [[J1[j][e] + J0[j][e] - 1 for e in EN] for j in range(6)],
           "tr": [[total(J1[j], t) - total(VS[j], t) + 1, total(VS[j], t) - 1 - total(J1[j], t)] for j in range(6)],
           "open": [[1 - J0[j][e] for e in EN] for j in range(6)], "valid": valid}
    st, o = go(st, new, drop=("rF", "m1", "m0", "mv", "eat"))
    t = net.t
    st["open"] = o["open"]
    st["bad"] = [total(o["cf"][j], t) + total(o["tr"][j], t) + 1 - o["valid"][j] for j in range(6)]

    # ---- connect: flood over blocks from block 0 along the open pairs
    ARC = [(u, w, e) for e, (u, w) in enumerate(PAIRS)] + [(w, u, e) for e, (u, w) in enumerate(PAIRS)]
    INB = [[i for i, (u, w, e) in enumerate(ARC) if w == b] for b in B]
    S = [[E(t, {}, 1 if b == 0 else 0) for b in B] for _ in range(6)]
    for i in range(24):
        m = [[step01(S[j][u] + A * (st["open"][j][e] - 1)) for u, w, e in ARC] for j in range(6)]
        st, o = go(st, {"m": m})
        t = net.t
        msg = [[b2(u) for u in o["m"][j]] for j in range(6)]
        S = [[total([msg[j][i] for i in INB[b]], t) + (1 if b == 0 else 0) for b in B] for j in range(6)]
    st, o = go(st, {"un": [[1 - S[j][b] for b in B] for j in range(6)]}, drop=("open",))
    t = net.t
    bad = [st["bad"][j] + total(o["un"][j], t) for j in range(6)]
    st, o = go(st, {"ok": [1 - bad[j] for j in range(6)]}, drop=("bad",))
    ok = o["ok"]

    # ---- end. Conforming candidates also need their first move to be safe.
    fo = st["fo"]
    st, o = go(st, {"g": ok[:2], "gc": [[ok[j] + ok[5] + fo[j] - 2, ok[j] + ok[4] - fo[j] - 1] for j in (2, 3)],
                    "s": ok[4:]})
    g = [o["g"][0], o["g"][1]] + [o["gc"][i][0] + o["gc"][i][1] for i in (0, 1)] + [o["s"][1], o["s"][0]]
    # first acceptable in order: 4 candidates, OUT, IN, then the last resort
    new = {"f": [g[i] - total(g[:i], net.t) for i in range(6)], "last": 1 - total(g, net.t) + st["free"][1] - 1}
    st, o = go(st, new, drop=("free",))
    f = o["f"]
    st, o = go(st, {"u": [f[i] + st["fo"][i] - 1 for i in range(4)] + [f[4], o["last"]]}, drop=("fo",))
    OUT = total(o["u"], net.t)
    dirs = {-10: 0, 1: 1, 10: 2, -1: 3}

    def heads(m, d):
        return total([st["H"][x] for x in X if MV[x][m] is not None and dirs[MV[x][m] - x] == d], net.t)
    o = net.step({}, {"z": [[heads(1, d) + OUT - 1, heads(0, d) - OUT] for d in range(4)]})
    net.step({}, {"out": [total(o["z"][d], net.t) for d in range(4)]})
    return net


if __name__ == "__main__":
    mats = to_matrices(build())
    nnz = sum(W.nnz + int((b != 0).sum()) for W, b in mats)
    print(f"layers {len(mats)}, nonzero weights {nnz:,}, widest {max(W.shape[0] for W, _ in mats)}")
    pickle.dump(mats, open("v6_net.pkl", "wb"))
