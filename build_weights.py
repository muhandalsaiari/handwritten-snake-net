"""Hand-written Snake network, version 3. No training, no data, no loss, no
gradients: every weight is set directly from the design below, like
Experiment 5 of github.com/RikiyaT/lm_understand-logs.

(Version 2, one hidden layer and a fixed buffer of 35, is in build_weights_v2.py.)

Network:  x (400) -> ReLU (1300) -> ReLU (500) -> logits (4); argmax = move.
Input x:  four one-hot blocks of 100: head cell, tail cell, food cell, length.
Moves:    0 up, 1 right, 2 down, 3 left.

Idea: the cells are numbered along a loop that visits every cell once (LOOP).
The body always lies in loop order, so every cell ahead of the head and before
the tail is free. Following the loop never crashes. A shortcut (jumping ahead
on the loop) is also safe if it lands before the tail and does not pass the
food. To avoid leaving holes the food can spawn in, a shortcut must also leave
a buffer of free cells before the tail: 5 + length / 2.

Layer 1:
  S[c,m]   400 units: head on c, move m lands before the tail and does not
           pass the food. Value = jump length + 1, else 0.
  P1, P2   800 units: P1 - P2 is a 0/1 step that says "the buffer is big
           enough" for head c and move m:  2*ahead - length - 2*(jump+5) > 0.
  L[c]     100 units: head on c. Backup that votes for the plain loop move.
Layer 2:
  Z[c,m]   400 units: ReLU(S + A*(P1 - P2) - A) for shortcuts, ReLU(S) for the
           plain loop move. So a shortcut keeps its value only if the buffer
           is big enough.
  B[c]     100 units: copy of L[c].
Output:    Z votes for its move with weight 2, B for the loop move with weight 1.

The loop shape and the buffer were picked by testing a few options in a
simulator (30 loop shapes, about 10 buffer rules, 40 games each); that is the
only use of play. Nothing was fitted.
"""
import numpy as np

N = 10
CELLS = N * N
A = 256  # gate size, bigger than any value a unit can carry
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]
B0, LEN_DIV = 5, 2  # buffer = 5 + length / 2

LOOP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 18, 17, 16, 15, 14, 13, 12, 11, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 39, 49, 59, 58, 48, 38, 37, 47, 57, 67,
        68, 69, 79, 78, 77, 87, 88, 89, 99, 98, 97, 96, 95, 94, 84, 85, 86, 76,
        66, 56, 46, 36, 35, 45, 55, 65, 75, 74, 64, 54, 44, 34, 33, 32, 31, 41,
        42, 43, 53, 63, 73, 83, 93, 92, 82, 72, 62, 52, 51, 61, 71, 81, 91, 90,
        80, 70, 60, 50, 40, 30, 20, 10]
assert sorted(LOOP) == list(range(CELLS))
POS = np.zeros(CELLS, dtype=int)
for i, cell in enumerate(LOOP):
    POS[cell] = i

# input blocks
H, T, F, LN = 0, CELLS, 2 * CELLS, 3 * CELLS
# layer 1 blocks
S, P1, P2, LB = 0, 4 * CELLS, 8 * CELLS, 12 * CELLS
HID1 = 13 * CELLS
# layer 2 blocks
Z, BB = 0, 4 * CELLS
HID2 = 5 * CELLS


def dist(a, b):
    """Steps forward along the loop from cell a to cell b."""
    return (POS[b] - POS[a]) % CELLS


def neighbour(c, m):
    r, k = divmod(c, N)
    r, k = r + MOVES[m][0], k + MOVES[m][1]
    return r * N + k if 0 <= r < N and 0 <= k < N else None


def build():
    W1 = np.zeros((4 * CELLS, HID1), dtype=np.float32)
    b1 = np.zeros(HID1, dtype=np.float32)
    W2 = np.zeros((HID1, HID2), dtype=np.float32)
    b2 = np.zeros(HID2, dtype=np.float32)
    W3 = np.zeros((HID2, 4), dtype=np.float32)

    for c in range(CELLS):
        # backup: always votes for the plain loop move
        W1[H + c, LB + c] = 1
        W2[LB + c, BB + c] = 1

        for m in range(4):
            n = neighbour(c, m)
            if n is None:
                continue  # off the board: these units stay at 0
            u = c * 4 + m
            jump = dist(c, n)

            # S: head gate + tail gate + food gate (with value) - 3A
            W1[H + c, S + u] = A
            for t in range(CELLS):
                if t != c and jump < dist(c, t):
                    W1[T + t, S + u] = A
            for f in range(CELLS):
                if f != c and jump <= dist(c, f):
                    W1[F + f, S + u] = A + jump + 1
            b1[S + u] = -3 * A

            # P1, P2: s = 2*ahead - length - 2*(jump + B0), only with head on c
            for p, extra in ((P1, 0), (P2, -1)):
                W1[H + c, p + u] = A
                for t in range(CELLS):
                    if t != c:
                        W1[T + t, p + u] = 2 * dist(c, t)
                for length in range(1, CELLS + 1):
                    W1[LN + length - 1, p + u] = -length * 2 // LEN_DIV
                b1[p + u] = -A - 2 * (jump + B0) + extra

            # Z: shortcuts need the buffer step; the loop move does not
            W2[S + u, Z + u] = 1
            if jump > 1:
                W2[P1 + u, Z + u] = A
                W2[P2 + u, Z + u] = -A
                b2[Z + u] = -A
            W3[Z + u, m] = 2
            if jump == 1:
                W3[BB + c, m] = 1
    return W1, b1, W2, b2, W3


if __name__ == "__main__":
    ws = build()
    total = sum(w.size for w in ws)
    used = sum(int((w != 0).sum()) for w in ws)
    print(f"weights: {total:,} total, {used:,} nonzero")
