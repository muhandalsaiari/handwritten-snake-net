# A Snake neural network with zero training

Every weight in this network was written by hand. No training, no data, no gradients.
It plays perfect games of Snake on a 10×10 board (the snake fills the whole board).

**Vibecoded:** I built this with Claude (Anthropic's AI) doing most of the coding. The idea
came from [lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs), where an AI writes
neural network weights directly instead of training them.

**Try it:** open `snake.html` in a modern browser. It's one 435 KB file and needs no internet.
(Tested on a PC. Not tried on a phone yet.)

## Results

| | Wins | Average moves to fill the board |
|---|---|---|
| v1: follow a loop + shortcuts | 1000 / 1000 | 2275 |
| v2: keep a gap before the tail | 1000 / 1000 | 1676 |
| v3: gap grows with length, better loop | 1000 / 1000 | 1586 |
| v4: sees the whole body | 1000 / 1000 | 1460 |
| **v5: 64 loops to choose from, time-aware paths** | **2000 / 2000** | **1241** |

- v5's network was checked against its plain written-out rule (`v5_rule.py`): same move on
  6,300 game moves and on 20,000 odd, tangled snakes, with no mismatches.
- The rows are the networks playing by themselves, on games not used to pick v5's settings.
  The v5 rule also won 30,000 of 30,000 games while its last settings were being chosen.
- That is strong evidence, not a proof. A very rare loss is still possible.
- v5 takes 2.6 ms per move in the browser on my PC (v4: 5.1 ms).

## How v5 plays

1. **Short snake (under 46):** find the shortest path to the food, counting body cells that will
   have emptied by the time the head gets there. Take it if the head could still reach the tail
   after eating. If not, stall by moving as far from the tail as possible.
2. **Long snake:** switch to one of 64 loops that visit every cell once, when the way ahead on it
   clears with 3 moves to spare.
3. **In order on a loop:** follow it, taking shortcuts that keep the body in order.

## How the network does that

- **Input (300 numbers):** where the head is, where the food is, and for every cell how many
  moves until the body leaves it.
- **94 layers, 902,066 nonzero weights.** All whole numbers, so the maths is exact.
- **Loop layers:** check all 64 loops at once.
- **Flood-fill layers:** each layer spreads "reached" one cell further. Counting layers gives
  distances.
- **Tracing layers:** walk the shortest path back from the food, one cell per layer.
- **One more flood:** checks the imagined snake (after eating) can still reach its tail.
- **Last layers:** pick the mode and output one move.

Every layer has a written purpose (see `build_v5.py`).

## Honest notes

- It only works on a 10×10 board.
- I haven't compared it head-to-head with trained Snake networks. The strongest one I found
  ([surt91](https://blog.schawe.me/en/perfect-snake.html)) also plays perfect 10×10 games but
  doesn't publish win rates or move counts, so I don't claim it's "the best".
- Settings (when to switch, how many loops, gap sizes) were picked by playing many simulated
  games with different values. The loops are random. No weight was trained.

## Files

| File | What it is |
|---|---|
| `snake.html` | v5, playable in a browser |
| `snake_v4.html` | v4, playable in a browser |
| `snake_v3.html` | v3, playable in a browser |
| `build_weights.py` | writes the v3 weights |
| `v4_rule.py` | v4's rule in plain Python |
| `build_v4.py` | writes the v4 weights |
| `test_v4.py` | checks the network against the rule, and lets it play |
| `v5_rule.py`, `loops_v5.json` | v5's rule in plain Python, and its 64 loops |
| `build_v5.py` | writes the v5 weights |
| `test_v5.py`, `stress_v5.py` | check the v5 network and let it play |
| `lab/` | the experiments used to pick v5's settings |
| `evaluate.py` | plays v3 for many games |
| `make_page.py`, `make_page_v4.py`, `make_page_v5.py` | pack the weights into the HTML pages |

## Run it yourself

```bash
pip install numpy scipy
python build_v5.py
python test_v5.py match 3
python make_page_v5.py
```

## License

MIT, see `LICENSE`.
