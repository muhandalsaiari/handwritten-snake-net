# A Snake neural network with zero training

Every weight in this network was written by hand. No training, no data, no gradients.
It plays perfect games of Snake on a 10×10 board (the snake fills the whole board).

**Vibecoded:** I built this with Claude (Anthropic's AI) doing most of the coding. The idea
came from [lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs), where an AI writes
neural network weights directly instead of training them.

**Try it:** open `snake.html` in a modern browser. It's one 79 KB file and needs no internet.
(Tested on a PC. Not tried on a phone yet.)

## Results

| | Wins | Average moves to fill the board |
|---|---|---|
| v1: follow a loop + shortcuts | 1000 / 1000 | 2275 |
| v2: keep a gap before the tail | 1000 / 1000 | 1676 |
| v3: gap grows with length, better loop | 1000 / 1000 | 1586 |
| v4: sees the whole body | 1000 / 1000 | 1460 |
| v5: 64 loops to choose from, time-aware paths | 2000 / 2000 | 1241 |
| v6: reshapes its loop as it goes | 2000 / 2000 | 1002 |
| **v7: free early play, plans for the next food** | **2000 / 2000** | **969** |

- v7's network was checked against its plain written-out rule (`v7_rule.py`): same move on
  2,806 game moves and 40,000 odd snakes, with no mismatches.
- The rows are the networks playing by themselves.
- A C++ copy of the v7 rule (`lab/v7.cpp`) won 200,000 of 200,000 games on seeds not used for
  tuning: 967.4 moves on average, 99.96% of them within 1200.
- That is strong evidence, not a proof. A very rare loss is still possible.
- v7 is a bigger network than v6, so each move takes longer: 5.5 ms in the browser on my PC
  (v6: 2.4 ms, measured back to back).

## How v7 plays

The board is 25 blocks of 2×2. In every cell the snake has two choices: turn inside the block, or
cross to the next block. If the blocks it joins form a tree, its path is a loop through every cell.

**Under length 20** it ignores the blocks: it takes the shortest path in any direction if it could
still reach its tail after eating. Then it folds into block driving, along a checked path or one of
8 fixed loops whose way ahead clears. After that:

1. **Find short paths to the food** (up to 24 moves), counting body cells that will have emptied
   by then. Take one if, after eating, the body still fits some loop through the whole board.
   Among those, pick the best mix of short and close to the remaining free cells (the next food).
2. **Otherwise** make a single move that keeps the body fitting some loop.
3. **Last resort:** cross to the next block if it's free.

## How the network does that

- **Input (300 numbers):** where the head is, where the food is, and for every cell how many
  moves until the body leaves it.
- **96 layers, 1,836,564 nonzero weights.** All whole numbers, so the maths is exact.
- **Flood layers:** each layer spreads "reached" one cell further. Counting layers gives distances.
- **Trace layers:** walk four candidate paths back from the food, one cell per layer.
- **Loop check:** count the joined blocks, then flood over the 25 blocks to see they can all still
  be connected.
- **Last layers:** pick the first acceptable option and output one move.

Every layer has a written purpose (see `build_v7.py`).

## Honest notes

- It only works on a 10×10 board.
- I don't claim it's the fastest Snake AI. Published 10×10 averages, as their authors report them (I haven't
  rerun them, and rules such as start length differ):
  - [Nomarcus/SnakeAI](https://github.com/Nomarcus/SnakeAI), Dynamic Loop: about 980 moves, 100% wins (algorithm).
    v7's average is lower, but I haven't run the two side by side, and I don't know its start length
    (ours starts at 1)
  - [slitherin](https://github.com/VonAlphaBisZulu/slitherin), Hamilton Cycle: 1092 moves, 100 of 100 wins (algorithm, searches each move)
  - [AlphaSnake](https://arxiv.org/abs/2211.09622), a trained network (AlphaZero): 944 of 1000 games won within 1200 moves.
    Not a fair comparison: their snake starts at length 2 (ours at 1), and I haven't run it.
- I found no other Snake network with hand-written weights, but I only did a short search.
- The block-tree idea isn't mine: see the cell tree algorithm in [twanvl/snake](https://github.com/twanvl/snake).
  The code and weights here are written separately.
- Settings (such as v7's length-20 switch, or v5's switch length) were picked by playing many simulated
  games with different values. v5's loops are random. No weight was trained.

## Files

| File | What it is |
|---|---|
| `snake.html` | v7, playable in a browser |
| `snake_v6.html` | v6, playable in a browser |
| `snake_v5.html` | v5, playable in a browser |
| `snake_v4.html` | v4, playable in a browser |
| `snake_v3.html` | v3, playable in a browser |
| `build_weights.py` | writes the v3 weights |
| `v4_rule.py` | v4's rule in plain Python |
| `build_v4.py` | writes the v4 weights |
| `test_v4.py` | checks the network against the rule, and lets it play |
| `v5_rule.py`, `loops_v5.json` | v5's rule in plain Python, and its 64 loops |
| `build_v5.py` | writes the v5 weights |
| `test_v5.py`, `stress_v5.py` | check the v5 network and let it play |
| `v6_rule.py` | v6's rule in plain Python |
| `build_v6.py` | writes the v6 weights (uses the builder in `build_v4.py` and `build_v5.py`) |
| `test_v6.py`, `stress_v6.py` | check the v6 network and let it play |
| `v7_rule.py`, `loops_v7.json` | v7's rule in plain Python, and its 8 loops |
| `build_v7.py` | writes the v7 weights |
| `test_v7.py`, `stress_v7.py` | check the v7 network and let it play |
| `lab/` | the experiments used to pick the settings |
| `evaluate.py` | plays v3 for many games |
| `make_page.py`, `make_page_v4.py`, `make_page_v5.py`, `make_page_v6.py`, `make_page_v7.py` | pack the weights into the HTML pages |

## Run it yourself

```bash
pip install numpy scipy
python build_v7.py
python test_v7.py match 3
python make_page_v7.py
```

## License

MIT, see `LICENSE`.
