"""Check the v5 network makes exactly the moves of v5_rule.py, then score it.

python test_v5.py match [games]   compare every move of rule-played games
python test_v5.py play [games] [seed]   let the network play by itself
"""
import pickle
import sys

import numpy as np

import v5_rule as R
from build_v5 import encode, forward
from play import play, report

mats = pickle.load(open("v5_net.pkl", "rb"))


def net_decide(body, food):
    out = forward(mats, encode(body, food))
    return int(np.argmax(out))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "match"
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    if mode == "match":
        rng = np.random.default_rng(11)
        states = bad = notone = 0
        for g in range(games):
            rec = []
            play(R.decide, rng, record=rec)
            for body, food, m in rec:
                out = forward(mats, encode(body, food))
                states += 1
                if sorted(np.round(out, 6).tolist()) != [0, 0, 0, 1]:
                    notone += 1
                if int(np.argmax(out)) != m:
                    bad += 1
                    if bad <= 5:
                        print("mismatch: len", len(body), "head", body[0], "food", food,
                              "rule", m, "net", np.round(out, 3).tolist())
        print(f"states {states}, mismatched moves {bad}, outputs not exactly one-hot {notone}")
    else:
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        rng = np.random.default_rng(seed)
        report(f"v5 network, {games} games", [play(net_decide, rng) for _ in range(games)])
