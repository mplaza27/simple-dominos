# Simple Dominos

A Cuban Dominos (Double-9) game engine paired with an ML strategy arena. Train, simulate, and compare AI strategies — then play against them in the browser.

## Game Rules

> For full mathematical analysis (combinatorics, branching factor, hidden information, skill vs luck), see [STATS.md](STATS.md).

- **55 tiles** (Double-9 set, pips 0-9)
- **4 players, 2 teams** — Team A (seats 0, 2) vs Team B (seats 1, 3)
- Each player gets 10 tiles; 15 remain hidden and out of play
- Player with the highest double opens (can play any tile)
- Match an open end or pass — no drawing from the hidden tiles
- Round ends when someone empties their hand (win) or all pass (stalemate — lowest pip sum wins)
- Points = opposing team's remaining pips

## Strategies

| Strategy | Approach |
|----------|----------|
| Random | Uniform random from valid moves |
| Greedy | Highest pip-sum tile first |
| Greedy Doubles | Doubles first, then highest pip-sum |
| Non-Greedy | Lowest pip-sum tile first |
| Late Game | Minimize remaining pips, save connectors |
| Never Passed | Maximize pip-value coverage to avoid passing |
| Pass Tracker | Infer opponent holdings from passes, exploit gaps |
| Partner Aware | Track partner's plays, cooperate on shared values |
| **RL Agent** | Neural network trained via PPO against all strategies + self-play |

## Project Structure

```
backend/
├── engine/          # Core game logic (types, tile, player, game)
├── strategies/      # 8 rule-based + 1 RL strategy
├── rl/              # Encoding, network, PPO trainer, ONNX export
├── simulation/      # Batch runner, Elo system, round-robin eval
└── tests/           # 127 pytest tests
frontend/public/
├── index.html       # Strategy battle viewer (animated Elo leaderboard)
├── play.html        # Interactive game UI
├── test-engine.html # JavaScript engine test suite
├── js/              # Game engine + strategies (vanilla JS)
└── data/            # Pre-computed simulation results (JSON)
scripts/
├── run_simulation.py  # Batch simulation → JSON
├── train_rl.py        # RL training (PPO, phased curriculum)
└── eval_rl.py         # Round-robin evaluation
```

## Setup

```bash
# Activate venv
source .venv/bin/activate

# Install dependencies
pip install pytest torch
```

## Quick Start

### Run tests

```bash
python -m pytest backend/tests/ -v
```

### Run a simulation (100k games)

```bash
python scripts/run_simulation.py --games 100000 --output frontend/public/data/results.json
```

### Train the RL agent

```bash
# Full training (v2 phased curriculum, 1M episodes across 3 phases)
python scripts/train_rl.py --phased --save-path models/domino_rl_v2.pt

# Quick test (10k episodes)
python scripts/train_rl.py --no-phased --episodes 10000

# Resume from checkpoint
python scripts/train_rl.py --phased --save-path models/domino_rl_v2.pt --resume
```

**Training phases (v2):**

| Phase | Episodes | Opponents | Self-Play |
|-------|----------|-----------|-----------|
| Curriculum | 200k | Graduated: weak → all 8 | 0% |
| Hard opponents | 300k | Top 4 only | 40% |
| Self-play | 500k | Top 4 + frozen self | 70% |

### Evaluate the RL agent

```bash
python scripts/eval_rl.py --model models/domino_rl_v2.pt --games-per-matchup 500
```

Outputs an Elo leaderboard and head-to-head win rate matrix using round-robin team matchups.

### Play in the browser

```bash
cd frontend/public && python3 -m http.server 8080
```

- **http://localhost:8080/index.html** — Strategy battle viewer (animated Elo leaderboard, game browser)
- **http://localhost:8080/play.html** — Play against AI opponents

## RL Agent

The RL agent uses a 4-layer MLP with residual connections, trained via PPO.

**State encoding (511 dims)** mirrors what a human player tracks:
- Tiles in hand and on the board
- Board end values and game state
- Per-player tile counts and pass history
- **Pip value tracking** — how many of each pip (0-9) have been played
- **End scarcity** — is a board end about to lock?
- **Per-seat play counts** — how many of each pip each player has played
- **Opponent inference** — which values opponents lack (from passes)

**Training pipeline:**
```
GameState → encode (511-dim) → DominoNet → masked policy + value
  → play game → GAE advantages → PPO clipped loss → gradient step
```

## Documentation

- [README.md](README.md) — This file. Project overview and quick start.
- [STATS.md](STATS.md) — Game statistics and mathematical analysis (combinatorics, information theory, skill vs luck).
- [PRD.md](PRD.md) — Product requirements, milestones, and RL design.
- [TODO.md](TODO.md) — Current work items, training operations, and setup tips.
- [CLAUDE.md](CLAUDE.md) — Code conventions and architecture rules.

## Architecture Notes

- `GameState` enforces hidden information — never exposes other players' hands or undealt tiles
- All strategies implement the `Strategy` ABC (`choose_action(state) → Action`)
- `rng` injection everywhere for reproducibility
- Frozen dataclasses with `slots=True` for immutable game state
- Frontend mirrors the Python engine in JavaScript for browser play
