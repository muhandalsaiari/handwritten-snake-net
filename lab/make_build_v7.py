"""Makes build_v7.py: v6's builder turned into a part ('core') plus v7_parts.py."""
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
s = open("build_v6.py").read()


def r(a, b, n=1):
    global s
    assert s.count(a) >= 1, a
    s = s.replace(a, b) if n == 0 else s.replace(a, b, n)


i = s.index('"""'); j = s.index('"""', i + 3) + 3
s = s[:i] + '''"""Hand-written Snake network, version 7 (the rule is in v7_rule.py).
No training, no gradients: every weight is set from the design below.

Input (300): head one-hot, food one-hot, age of every cell.
Every unit is ReLU(weights . previous layer + bias); all numbers are whole.
Three parts run side by side, then the last layers choose between them:

  core       v6's block driving: path floods, traced candidates, "does the body
             still fit some loop", plus a 10-step flood from each candidate's
             end that measures how close the remaining free cells are
  free_play  early play in all 4 directions with a tail check
  loops      which of 8 loops clears ahead, for folding into block driving
"""''' + s[j:]
r("import v6_rule as R\n", "import v7_rule as V\n")
r("from build_v5 import b2, eq0, lin, total\n", "from build_v5 import b2, compare, eq0, lin, total\n")
r("from v6_rule import BL, EDGE, MV, PAIRS, PRED\n", "from v7_rule import BL, EDGE, MV, NB, NL, PAIRS, POS, PRED\n")
r("C, D = 100, R.DMAX\n", "C, D, BIG = 100, V.DMAX, 20000\n")
r('CAND = [', '''M4 = range(4)
PAIR4 = [(x, m) for x in range(100) for m in M4 if NB[x][m] is not None]
FROM4 = [[None] * 4 for _ in range(100)]
for _x, _m in PAIR4:
    FROM4[NB[_x][_m]][_m] = _x
CAND = [''')
a = s.index("def build():"); b = s.index("    # 1: age thresholds")
s = s[:a] + '''def core(net, H, F, a):
    def go(st, new, drop=()):
        k, n = yield ({k: v for k, v in st.items() if k not in drop}, new)
        return k, n

''' + s[b:]
s = s.replace("st, o = go(", "st, o = yield from go(")
# which body cells did not move IN / OUT
r('    st, o = yield from go(st, new, drop=("blk",))', '''    new["nk"] = [st["occ"][x] - st["H"][x] - total([bm[(x, q)] for q in (0, 1) if (x, q) in bm], net.t) for x in X]
    Lx = total(st["occ"], net.t)
    new["L"], new["gk"] = Lx, [Lx - V.K + 1, Lx - V.K]
    st, o = yield from go(st, new, drop=("blk",))
    st["nk"], st["L"], st["geK"] = o["nk"], o["L"], b2(o["gk"])''')
r('    st["occ"] = st["occ"]\n', "")
r('    new = {"tl": [[', '    new = {"nkr": step01(total(st["nk"], t)), "tl": [[')
r('    al = {s: (b2', '    st["nkr"] = b2(o["nkr"])\n    al = {s: (b2')
# imagined ages for the next-food flood; keep aF for path lengths
r('''           "fo": [[st["H"][x] + ON[j]''', '''           "vp": {j: {s: L - (D - 1) + JJ[j][s] + A * (ON[j][s] - 1) for s in STEPS} for j in range(4)},
           "vo": {n: [st["a"][x] - D + st["aF"][n] + A * (st["rF"][n] - 1) for x in X] for n in "pc"},
           "fo": [[st["H"][x] + ON[j]''')
r('drop=("a", "occ", "aF"))', 'drop=("a", "occ"))')
r('    st["fo"] = [total(v, t) for v in o["fo"]]', '''    vage = {j: [total([o["vp"][j][(x, q)] for q in (0, 1) if (x, q) in o["vp"][j]], t)
                + o["vo"][CAND[j][0]][x] for x in X] for j in range(4)}
    st["fo"] = [total(v, t) for v in o["fo"]]''')
r('''           "ae": ae,
           "vis"''', '''           "ae": ae,
           "nb": {n: [ao[n][x] + st["nk"][x] - 1 for x in X] for n in "pc"},
           "op": {j: [[z - 1, z - 2] for z in vage[j]] for j in range(4)},
           "fz": {j: [1 - vage[j][x] - st["F"][x] for x in X] for j in range(4)}, "N": st["F"],
           "vis"''')
r('drop=("bm", "F"))', 'drop=("bm", "F", "nk"))')
r('''    k, ae = o["k"], o["ae"]
''', '''    k, ae = o["k"], o["ae"]
    st["nb"] = {n: total(o["nb"][n], t) for n in "pc"}
    st["op"], st["fz"], st["N"] = o["op"], o["fz"], o["N"]
''')
r('"mv", "eat"))', '"mv", "eat", "nb"))')
r('"open": [[1 - J0', '"nb": st["nb"], "open": [[1 - J0')
r('+ 1 - o["valid"][j] for j in range(6)]', '+ 1 - o["valid"][j]\n                 + (o["nb"][CAND[j][0]] if j < 4 else 0) for j in range(6)]')
# next-food flood alongside the connect flood
r('''    for i in range(24):
        m = [[''', '''    op, fz = st.pop("op"), st.pop("fz")
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
        m = [[''')
r('        st, o = yield from go(st, {"m": m})\n        t = net.t\n', '''        st, o = yield from go(st, {"m": m, **new})
        t = net.t
        if i <= V.FUT:
            fz, Rp, acc, op = o["fz"], o["R"], o["acc"], o["op"]
            N = {j: [b2(u) for u in o["U"][j]] for j in range(4)}
        elif i == V.FUT + 1:
            st["fs"] = [total(o["fs"][j], t) for j in range(4)]
''')
a = s.index('    st, o = yield from go(st, {"ok": [1 - bad[j]')
s = s[:a] + '''    de = st["aF"]["p"] - st["aF"]["c"]      # how much longer the conforming path is
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

''' + open("lab/v7_parts.py").read().lstrip("\n")
open("build_v7.py", "w").write(s)
