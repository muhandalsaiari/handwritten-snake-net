"""Version 4 rule, written plainly, using only steps a network can do:
flood fills, tracing a path one cell at a time, and simple comparisons.
The network in build_v4.py must make exactly the same moves.

State the network sees: head cell, food cell, and the AGE of every cell
(0 = free, 1 = tail ... L = head: moves until the cell is free again).

  1. Long snake (length >= K) lying in loop order: the v3 rule (provably safe).
  2. Long snake not in order, but the loop ahead clears in time: loop move.
  3. Otherwise free play:
     a. Flood-fill from the head to find the shortest path to the food.
     b. Imagine following it; after eating, can the head still reach the
        tail? If yes, take the first step of the path.
     c. If not (or no path): stall, preferring the loop move, among moves
        that keep the tail reachable.
"""
from collections import deque

import build_weights as v3
from build_weights import CELLS, N, dist, neighbour

K = 20
MARGIN = 1
FAR = 999
DMAX = 64  # floods spread at most this many cells (one network layer each)
STALL = "farthest"  # "loop" circled forever in 1 of 100 games


def ages_of(body):
    a = [0] * CELLS
    L = len(body)
    for i, c in enumerate(body):
        a[c] = L - i
    return a


def flood(start, passable):
    """Distances from start over cells where passable(cell) is true."""
    d = [FAR] * CELLS
    d[start] = 0
    q = deque([start])
    while q:
        x = q.popleft()
        for m in range(4):
            y = neighbour(x, m)
            if y is not None and d[y] == FAR and passable(y) and d[x] < DMAX:
                d[y] = d[x] + 1
                q.append(y)
    return d


def tail_reachable(a):
    """Can the head reach the tail through free cells?"""
    L = sum(1 for v in a if v)
    if L <= 2:
        return True
    head = a.index(L)
    tail = a.index(1)
    return flood(head, lambda y: a[y] <= 1)[tail] < FAR


def moved(a, n, eat):
    """Ages after the head moves to n."""
    L = sum(1 for v in a if v)
    b = [max(v - 1 + eat, 0) if v else 0 for v in a]
    b[n] = L + eat
    return b


def loop_ok(a, head):
    """Following the loop from here, every body cell is gone in time."""
    return all(v <= dist(head, x) - MARGIN for x, v in enumerate(a) if v and x != head)


def ordered(a):
    total = 0
    for x in range(CELLS):
        if a[x] >= 1:
            for m in range(4):
                y = neighbour(x, m)
                if y is not None and a[y] == a[x] + 1:
                    total += dist(x, y)
    return total < CELLS


def loop_move(head):
    for m in range(4):
        n = neighbour(head, m)
        if n is not None and dist(head, n) == 1:
            return m


def v3_move(head, tail, food, L):
    ahead = dist(head, tail) if head != tail else CELLS
    best, pick = -1, loop_move(head)
    for m in range(4):
        n = neighbour(head, m)
        if n is None:
            continue
        j = dist(head, n)
        if j == 1 or head == tail or j > dist(head, food) or j >= ahead:
            continue
        if 2 * ahead - L - 2 * (j + v3.B0) <= 0:
            continue
        if j > best:
            best, pick = j, m
    return pick


def path_to_food(a, head, food):
    """Shortest path head -> food, traced back from the food: from each cell
    step to the first neighbour (in move order) one step closer to the head."""
    d = flood(head, lambda y: a[y] <= 1)
    if d[food] >= FAR:
        return None
    path = [food]
    while d[path[-1]] > 1:
        x = path[-1]
        for m in range(4):
            y = neighbour(x, m)
            if y is not None and d[y] == d[x] - 1:
                path.append(y)
                break
    return path[::-1]  # path[0] is next to the head


def decide(body, food):
    a = ages_of(body)
    head, L = body[0], len(body)
    if L >= K and ordered(a):
        return v3_move(head, body[-1], food, L)
    if L >= K and loop_ok(a, head):
        return loop_move(head)

    path = path_to_food(a, head, food)
    if path is not None:
        k = len(path)
        b = [max(v - (k - 1), 0) if v else 0 for v in a]  # old body after k moves, eating at the end
        for i, p in enumerate(path, start=1):
            b[p] = max(L + 1 - (k - i), 0)  # early path cells may be gone again
        if tail_reachable(b):
            return next(m for m in range(4) if neighbour(head, m) == path[0])

    safe = []
    for m in range(4):
        n = neighbour(head, m)
        if n is None or a[n] >= 2:
            continue
        b = moved(a, n, 1 if n == food else 0)
        if tail_reachable(b):
            safe.append((m, n, b))
    if not safe:
        legal = [m for m in range(4) if neighbour(head, m) is not None and a[neighbour(head, m)] <= 1]
        return loop_move(head) if not legal or loop_move(head) in legal else legal[0]
    if STALL == "loop":
        for m, n, b in safe:
            if m == loop_move(head):
                return m
        return safe[0][0]
    # Farthest from the tail: the same flood as the safety check, measured
    # at the tail the snake would have after the move.
    best = None
    for m, n, b in safe:
        L2 = sum(1 for v in b if v)
        dt = flood(n, lambda y: b[y] <= 1)[b.index(1)] if L2 > 2 else 0
        if best is None or dt > best[0]:
            best = (dt, m)
    return best[1]
