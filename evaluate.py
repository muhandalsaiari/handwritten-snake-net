"""Score the hand-written network (version 3) on real Snake games. The game
rules here are separate from the network: it only sees head, tail, food and
length.

Usage: python evaluate.py [games] [seed]
"""
import sys
from collections import deque

import numpy as np

from build_weights import CELLS, F, LN, MOVES, N, T, build

W1, b1, W2, b2, W3 = build()


def policy(head, tail, food, length):
    # Same as the full forward pass with one-hot x: only the 4 rows x switches
    # on are added, and only the nonzero layer-1 units feed layer 2.
    h1 = np.maximum(W1[head] + W1[T + tail] + W1[F + food] + W1[LN + length - 1] + b1, 0)
    nz = np.flatnonzero(h1)
    h2 = np.maximum(h1[nz] @ W2[nz] + b2, 0)
    return int(np.argmax(h2 @ W3))


def play(rng, max_steps=200_000):
    start = int(rng.integers(CELLS))
    body = deque([start])
    occupied = {start}

    def new_food():
        free = [c for c in range(CELLS) if c not in occupied]
        return int(rng.choice(free)) if free else None

    food = new_food()
    for step in range(1, max_steps + 1):
        m = policy(body[0], body[-1], food, len(body))
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


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rng = np.random.default_rng(seed)
    results = [play(rng) for _ in range(games)]
    ends = {}
    for end, _, _ in results:
        ends[end] = ends.get(end, 0) + 1
    steps = [r[2] for r in results]
    print(f"games: {games}  endings: {ends}")
    print(f"mean steps to fill the board: {np.mean(steps):.0f} "
          f"(best {min(steps)}, worst {max(steps)})  worst length: {min(r[1] for r in results)}")
