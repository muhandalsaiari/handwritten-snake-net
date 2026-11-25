"""Hand-written Snake network. No training, no data, no loss, no gradients.

Every weight below is set directly from a design, like Experiment 5 of
github.com/RikiyaT/lm_understand-logs.

Network:  x (300) -> ReLU hidden (500) -> logits (4), argmax picks the move.
Input x:  three 10x10 one-hot planes: head, tail, food.
Moves:    0 up, 1 right, 2 down, 3 left.

Design: number the cells along a loop that visits every cell once. The body
always sits in loop order, so every cell ahead of the head and before the tail
is free. Following the loop can never crash. The network may also jump ahead
along the loop, as long as the jump lands before the tail and does not pass
the food. That keeps the order intact, so it is still safe, but much faster.

Hidden units 0..399:   one per (head cell, move). Fires only if the head is on
                       that cell and the jump is safe. Value = jump length + 1.
Hidden units 400..499: one per head cell. Adds 0.5 to the plain loop move, so
                       there is always a legal fallback.
"""
import numpy as np

N = 10
CELLS = N * N
A = 256  # gate size: bigger than any move score, so one failed condition
         # always pushes a unit to 0 or below
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def make_loop():
    """Column 0 is the way back up; the other columns are swept row by row."""
    path = [(0, 0)]
    for r in range(N):
        cols = range(1, N) if r % 2 == 0 else range(N - 1, 0, -1)
        path += [(r, c) for c in cols]
    path += [(r, 0) for r in range(N - 1, 0, -1)]
    assert len(set(path)) == CELLS
    return path


LOOP = make_loop()
POS = np.zeros(CELLS, dtype=int)
for i, (r, c) in enumerate(LOOP):
    POS[r * N + c] = i


def dist(a, b):
    """Steps forward along the loop from cell a to cell b."""
    return (POS[b] - POS[a]) % CELLS


def neighbour(c, m):
    r, k = divmod(c, N)
    r, k = r + MOVES[m][0], k + MOVES[m][1]
    return r * N + k if 0 <= r < N and 0 <= k < N else None


def forward_steps():
    """STEPS[n, f]: fewest moves from n to f if every move goes forward on the
    loop and never passes f. Those cells are all free, so any such path is
    safe. Worked out by hand-style counting, not learned."""
    steps = np.zeros((CELLS, CELLS), dtype=int)
    for f in range(CELLS):
        for x in sorted(range(CELLS), key=lambda x: dist(x, f)):
            if x == f:
                continue
            best = CELLS
            for m in range(4):
                n = neighbour(x, m)
                if n is not None and n != x and dist(x, n) <= dist(x, f):
                    best = min(best, 1 + steps[n, f])
            steps[x, f] = best
    return steps


STEPS = forward_steps()


def build(mode="old", buffer=35):
    """mode "old": score = jump length. mode "fast": score = fewest steps to
    the food (tested: no faster, so "old" is the default). buffer 35 was
    picked from a 40-game sweep (buffer 0: 2294 steps, 30-40: about 1710)."""
    H, T, F = 0, CELLS, 2 * CELLS
    W1 = np.zeros((3 * CELLS, 5 * CELLS), dtype=np.float32)
    b1 = np.full(5 * CELLS, -3.0 * A, dtype=np.float32)
    # Each live unit: head gate A + tail gate A + food weight (A + value) - 3A
    # = value > 0 when all three gates pass, and <= 0 when any fails.
    W2 = np.zeros((5 * CELLS, 4), dtype=np.float32)

    for c in range(CELLS):
        for m in range(4):
            n = neighbour(c, m)
            u = c * 4 + m
            if n is None:
                continue  # unit stays dead: bias -3A and no inputs
            jump = dist(c, n)
            W1[H + c, u] = A
            for t in range(CELLS):
                # Lands before the tail. A shortcut (jump > 1) must also leave
                # `buffer` free cells before the tail, so fewer holes form.
                need = jump if jump == 1 else jump + buffer
                if t != c and need < dist(c, t):
                    W1[T + t, u] = A
            for f in range(CELLS):
                if f != c and jump <= dist(c, f):
                    # Gate: does not pass the food. Value: how good the move is.
                    if mode == "fast":
                        # Fewest steps first; on a tie, the smaller jump.
                        W1[F + f, u] = A + 2 * (CELLS - STEPS[n, f]) - jump / CELLS
                    else:
                        W1[F + f, u] = A + jump + 1
            W2[u, m] = 1.0

            if jump == 1:  # this move is the plain loop move
                W2[4 * CELLS + c, m] = 0.5
        W1[H + c, 4 * CELLS + c] = 1.0
        b1[4 * CELLS + c] = 0.0
    return W1, b1, W2


if __name__ == "__main__":
    W1, b1, W2 = build()
    np.savez("snake_net.npz", W1=W1, b1=b1, W2=W2)
    total = W1.size + b1.size + W2.size
    used = int((W1 != 0).sum() + (b1 != 0).sum() + (W2 != 0).sum())
    print(f"weights: {total:,} total, {used:,} nonzero")
