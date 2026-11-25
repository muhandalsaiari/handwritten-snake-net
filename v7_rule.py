"""Version 7 rule, written plainly. build_v7.py turns it into weights.

Builds on v6 (blocks of 2x2, IN / OUT moves, "extendable" bodies):
  * Short (length < K): free play. Shortest path in all 4 directions, if the
    tail can still be reached after eating; else stall far from the tail.
  * Long, body not made of IN/OUT moves yet: fold into block driving: a checked
    IN/OUT path to the food, else one of 8 loops whose way ahead clears.
  * Otherwise v6's four candidate paths, but the best one is chosen by
    length + 1.5 x (average distance to the remaining free cells afterwards).
"""
import json
import os

import v6_rule as B
from v6_rule import BL, C, EDGE, FAR, FREE, MV, PAIRS, PRED, allowed_moves, extendable, trace

K, DMAX, FUT, MARGIN = 20, 24, 10, 4
B.DMAX = DMAX
NB = [[c - 10 if c > 9 else None, c + 1 if c % 10 < 9 else None,
       c + 10 if c < 90 else None, c - 1 if c % 10 else None] for c in range(C)]
NL = [[y for y in r if y is not None] for r in NB]
LOOPS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "loops_v7.json")))
POS = [[o.index(c) for c in range(C)] for o in LOOPS]
NXT = [[o[(p[c] + 1) % C] for c in range(C)] for o, p in zip(LOOPS, POS)]


def ages_of(body):
    a = [0] * C
    for i, c in enumerate(body):
        a[c] = len(body) - i
    return a


def flood4(h, a, timed):
    d = [FAR] * C
    d[h] = 0
    q = [h]
    for x in q:
        k = d[x] + 1
        if k > DMAX:
            break
        for y in NL[x]:
            if d[y] == FAR and a[y] <= (k if timed else 1):
                d[y] = k
                q.append(y)
    return d


def tail_dist(body):
    return 0 if len(body) <= 2 else flood4(body[0], ages_of(body), False)[body[-1]]


def step(body, n, food):
    return [n] + (body if n == food else body[:-1])


def after(body, path):
    return (path[::-1] + body)[:len(body) + 1]


def free_path(body, food, a):
    d = flood4(body[0], a, True)
    if d[food] < FAR:
        for last in (0, 1):
            p = [food]
            while d[p[-1]] > 1:
                c = [y for y in NL[p[-1]] if d[y] == d[p[-1]] - 1]
                p.append(c[-1] if last else c[0])
            p.reverse()
            if tail_dist(after(body, p)) < FAR:
                return p[0]
    return None


def stall(body, food, a):
    best = None
    for n in NL[body[0]]:
        if a[n] <= 1:
            t = tail_dist(step(body, n, food))
            if t < FAR and (best is None or t > best[0]):
                best = (t, n)
    if best:
        return best[1]
    return next((n for n in NL[body[0]] if a[n] <= 1), NL[body[0]][0])


def block_driving(body):
    return all(body[i - 1] in MV[body[i]] for i in range(1, len(body)))


def fits(body):
    return block_driving(body) and extendable(body)


def future(body):
    """Sum over free cells of their IN/OUT distance (at most FUT + 1)."""
    a = ages_of(body)
    B.DMAX = FUT
    d = B.flood(body[0], a, FREE)
    B.DMAX = DMAX
    return sum(min(d[x], FUT + 1) for x in range(C) if a[x] == 0)


def clears(body, a, j, margin):
    p = POS[j]
    return all(a[x] + margin <= (p[x] - p[body[0]]) % C for x in body[1:])


def next_cell(body, food):
    L, h, a = len(body), body[0], ages_of(body)
    if L < K:
        n = free_path(body, food, a)
        return n if n is not None else stall(body, food, a)
    if not block_driving(body):
        d = B.flood(h, a, FREE)
        if d[food] < FAR:
            for last in (0, 1):
                p = trace(d, food, FREE, last)
                if fits(after(body, p)):
                    return p[0]
        for j in range(len(LOOPS)):
            if clears(body, a, j, MARGIN):
                return NXT[j][h]
        if not any(a[n] <= 1 and tail_dist(step(body, n, food)) < FAR for n in NL[h]):
            for j in range(len(LOOPS)):
                if clears(body, a, j, 0):
                    return NXT[j][h]
        n = free_path(body, food, a)
        return n if n is not None else stall(body, food, a)
    free = [n is not None and a[n] <= 1 for n in MV[h]]
    safe = [free[t] and fits(step(body, MV[h][t], food)) for t in (0, 1)]
    best = None
    for conf, ok in enumerate((FREE, allowed_moves(body))):
        d = B.flood(h, a, ok)
        if d[food] < FAR:
            for last in (0, 1):
                p = trace(d, food, ok, last)
                b = after(body, p)
                if fits(b) and (not conf or safe[MV[h].index(p[0])]):
                    s = 2 * max(1, 99 - L) * len(p) + 3 * future(b)
                    if best is None or s < best[0]:
                        best = (s, p[0])
    if best:
        return best[1]
    for t in (1, 0):
        if safe[t]:
            return MV[h][t]
    return MV[h][1] if free[1] else MV[h][0]


def decide(body, food):
    return NB[body[0]].index(next_cell(list(body), food))
