
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
