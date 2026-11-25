"""Version 6 rule, written plainly. build_v6.py turns it into weights.

The board is 25 blocks of 2x2. In every cell the snake has only two moves:
IN (turn inside the block) or OUT (cross to the next block). If the blocks it
joins form a tree, its path is a loop through every cell. So the body is
"extendable" when some tree of all 25 blocks still fits it:
  * no pair of blocks is both joined and kept apart by the body,
  * joined pairs = visited blocks - 1 (no ring),
  * the pairs not kept apart still connect all 25 blocks.

Each move:
  1. Four candidate shortest paths to the food (at most DMAX moves; body cells
     open as they empty): plain and conforming flood, two tie-breaks each.
     Take the first whose body after eating is extendable. A conforming one
     must also stay extendable after its first step (it may change later).
  2. Else OUT, else IN, if the body after that one move is extendable.
  3. Else OUT if free, else IN.
"""
C, DMAX, FAR = 100, 20, 999


def _moves(c):
    y, x = divmod(c, 10)
    top, left = y % 2 == 0, x % 2 == 0
    if top and left:
        i, o = (0, 1), (-1, 0)
    elif top:
        i, o = (1, 0), (0, 1)
    elif not left:
        i, o = (0, -1), (1, 0)
    else:
        i, o = (-1, 0), (0, -1)
    oy, ox = y + o[0], x + o[1]
    return ((y + i[0]) * 10 + x + i[1],
            oy * 10 + ox if 0 <= oy < 10 and 0 <= ox < 10 else None)


MV = [_moves(c) for c in range(C)]                    # (IN cell, OUT cell or None)
BL = [(c // 20) * 5 + (c % 10) // 2 for c in range(C)]  # block of a cell
PRED = [[] for _ in range(C)]                         # (cell, 0 IN / 1 OUT) leading here
for _c in range(C):
    for _t in (0, 1):
        if MV[_c][_t] is not None:
            PRED[MV[_c][_t]].append((_c, _t))
_be = {}
EDGE = [None if MV[c][1] is None else _be.setdefault(tuple(sorted((BL[c], BL[MV[c][1]]))), len(_be))
        for c in range(C)]                            # block pair that this cell's move decides
PAIRS = list(_be)                                     # 40 neighbouring block pairs


def decisions(body):
    """Pairs joined (d1) and kept apart (d0) by the body; visited blocks."""
    d1, d0 = set(), set()
    for i in range(1, len(body)):
        c = body[i]
        if EDGE[c] is not None:
            if body[i - 1] == MV[c][1]:
                d1.add(EDGE[c])
            elif body[i - 1] == MV[c][0]:
                d0.add(EDGE[c])
    return d1, d0, {BL[c] for c in body}


def extendable(body):
    d1, d0, vis = decisions(body)
    if d1 & d0 or len(d1) != len(vis) - 1:
        return False
    seen, grew = {BL[body[0]]}, True
    while grew:
        grew = False
        for e, (u, w) in enumerate(PAIRS):
            if e not in d0 and (u in seen) != (w in seen):
                seen |= {u, w}
                grew = True
    return len(seen) == 25


def allowed_moves(body):
    d1, d0, vis = decisions(body)
    ok = [[True, MV[c][1] is not None] for c in range(C)]
    for c in range(C):
        e = EDGE[c]
        if e is not None:
            ok[c][0] = e not in d1
            ok[c][1] = e not in d0 and (e in d1 or BL[MV[c][1]] not in vis)
    return ok


def flood(head, age, ok):
    d = [FAR] * C
    d[head] = 0
    q = [head]
    for x in q:
        k = d[x] + 1
        if k > DMAX:
            break
        for t in (0, 1):
            y = MV[x][t]
            if y is not None and ok[x][t] and d[y] == FAR and age[y] <= k:
                d[y] = k
                q.append(y)
    return d


def trace(d, food, ok, last):
    p = [food]
    while d[p[-1]] > 1:
        c = [y for y, t in PRED[p[-1]] if ok[y][t] and d[y] == d[p[-1]] - 1]
        p.append(c[-1] if last else c[0])
    return p[::-1]


FREE = [[True, MV[c][1] is not None] for c in range(C)]


def choose(body, food):
    """0 = IN, 1 = OUT"""
    L, h = len(body), body[0]
    age = [0] * C
    for i, c in enumerate(body):
        age[c] = L - i
    free = [n is not None and age[n] <= 1 for n in MV[h]]
    safe = [free[t] and extendable([MV[h][t]] + body[:L if MV[h][t] == food else L - 1]) for t in (0, 1)]
    for conf, ok in enumerate((FREE, allowed_moves(body))):
        d = flood(h, age, ok)
        if d[food] < FAR:
            for last in (0, 1):
                p = trace(d, food, ok, last)
                t = MV[h].index(p[0])
                if extendable((p[::-1] + body)[:L + 1]) and (safe[t] or not conf):
                    return t
    for t in (1, 0):
        if safe[t]:
            return t
    return 1 if free[1] else 0


def decide(body, food):
    body = list(body)
    h = body[0]
    n = MV[h][choose(body, food)]
    return {-10: 0, 1: 1, 10: 2, -1: 3}[n - h]
