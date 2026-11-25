"""Shared Snake game for testing any policy decide(body, food) -> move."""
from collections import deque

from build_weights import CELLS, MOVES, N


def play(decide, rng, max_steps=50_000, record=None):
    start = int(rng.integers(CELLS))
    body = deque([start])
    occupied = {start}

    def new_food():
        free = [c for c in range(CELLS) if c not in occupied]
        return int(rng.choice(free)) if free else None

    food = new_food()
    for step in range(1, max_steps + 1):
        m = decide(body, food)
        if record is not None:
            record.append((list(body), food, m))
        r, c = divmod(body[0], N)
        r, c = r + MOVES[m][0], c + MOVES[m][1]
        if not (0 <= r < N and 0 <= c < N):
            return "wall", len(body), step
        n = r * N + c
        eating = n == food
        if not eating:
            occupied.discard(body.pop())
        if n in occupied:
            return "body", len(body), step
        body.appendleft(n)
        occupied.add(n)
        if eating:
            food = new_food()
            if food is None:
                return "win", len(body), step
    return "timeout", len(body), max_steps


def report(name, results):
    ends = {}
    for e, _, _ in results:
        ends[e] = ends.get(e, 0) + 1
    steps = [r[2] for r in results if r[0] == "win"]
    mean = f"{sum(steps) / len(steps):.0f}" if steps else "n/a"
    print(f"{name}: {ends}  mean moves (wins) {mean}", flush=True)
