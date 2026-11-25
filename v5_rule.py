"""Version 5 rule, written plainly, using only steps a network can do.
build_v5.py turns it into weights; test_v5.py checks every move matches.

State: head, food, and the AGE of every cell (0 free, 1 tail ... L head).

Short snake (length < K), and whenever nothing below applies:
  a. Time-aware flood from the head: a body cell may be entered once it has
     emptied. Trace the shortest path back from the food.
  b. Take it if, after eating, the head could still reach the tail.
  c. Otherwise stall: the tail-safe move farthest from the tail.
Long snake (length >= K), with 64 loops that each visit every cell once:
  1. Body already in order along some loop: follow it, with safe shortcuts.
  2. Else some loop ahead clears with 3 moves to spare: follow it.
  3. Else, if no move is tail-safe but some loop ahead clears: follow that.
"""
import json
import os

import build_weights as bw

C = 100
K, MARGINS, BUF, DMAX = 46, (3,), 10, 24
FAR = 999
NB = [[bw.neighbour(x, m) for m in range(4)] for x in range(C)]
NL = [[y for y in r if y is not None] for r in NB]
LOOPS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "loops_v5.json")))
POS = [[o.index(c) for c in range(C)] for o in LOOPS]
TAB = [[[(p[y] - p[x]) % C for y in range(C)] for x in range(C)] for p in POS]
NXT = [[next(y for y in NL[x] if t[x][y] == 1) for x in range(C)] for t in TAB]


def ages_of(body):
    a = [0] * C
    for i, c in enumerate(body):
        a[c] = len(body) - i
    return a


def flood(start, a, timed=False):
    d = [FAR] * C
    d[start] = 0
    q = [start]
    for x in q:
        k = d[x] + 1
        if k > DMAX:
            break
        for y in NL[x]:
            if d[y] == FAR and a[y] <= (k if timed else 1):
                d[y] = k
                q.append(y)
    return d


def tail_dist(a, head, L):
    return 0 if L <= 2 else flood(head, a)[a.index(1)]


def ordered(body, t):
    return sum(t[body[i + 1]][body[i]] for i in range(len(body) - 1)) < C


def clears(a, body, t, margin):
    D = t[body[0]]
    return all(a[x] + margin <= D[x] for x in body[1:])


def loop_shortcut(j, body, food):
    h, L = body[0], len(body)
    D = TAB[j][h]
    ahead, best, pick = D[body[-1]], 1, NXT[j][h]
    for n in NL[h]:
        if best < D[n] <= D[food] and 2 * ahead - 2 * D[n] - 2 * BUF - L > 0:
            best, pick = D[n], n
    return pick


def next_cell(body, food):
    L, h = len(body), body[0]
    a = ages_of(body)
    if L >= K:
        for j, t in enumerate(TAB):
            if ordered(body, t):
                return loop_shortcut(j, body, food)
        for margin in MARGINS:
            for j, t in enumerate(TAB):
                if clears(a, body, t, margin):
                    return NXT[j][h]
    safe = []
    for n in NL[h]:
        if a[n] <= 1:
            e = int(n == food)
            b = [v - 1 + e if v else 0 for v in a]
            b[n] = L + e
            t = tail_dist(b, n, L + e)
            if t < FAR:
                safe.append((t, n))
    if L >= K and not safe:
        for j, t in enumerate(TAB):
            if clears(a, body, t, 0):
                return NXT[j][h]
    d = flood(h, a, True)
    if d[food] < FAR:
        p = [food]
        while d[p[-1]] > 1:
            p.append(next(y for y in NL[p[-1]] if d[y] == d[p[-1]] - 1))
        p.reverse()
        k = len(p)
        b = [max(v - k + 1, 0) for v in a]
        for i, c in enumerate(p, 1):
            b[c] = max(L + 1 - k + i, 0)
        if tail_dist(b, food, L + 1) < FAR:
            return p[0]
    if safe:
        return max(safe, key=lambda s: s[0])[1]  # first one on ties
    if a[NXT[0][h]] <= 1:
        return NXT[0][h]
    return next((n for n in NL[h] if a[n] <= 1), NXT[0][h])


def decide(body, food):
    return NB[body[0]].index(next_cell(list(body), food))
