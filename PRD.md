# PRD: Cuban Dominos — ML Strategy Arena

## 1. Overview

A Cuban Dominos (Double-9) game engine paired with a machine learning platform to train, simulate, and compare AI strategies. Users can watch strategies compete in bulk simulations and play interactive games against any AI opponent — all from a single web UI deployable to GitHub Pages.

---

## 2. Game Rules — Cuban Dominos (Double-9)

### Tiles
- **55 unique tiles**: every combination (a, b) where 0 ≤ a ≤ b ≤ 9
- Tiles are unordered pairs — (3, 7) and (7, 3) are the same tile

### Players & Dealing
| Mode | Players | Teams | Tiles Dealt | Undealt (Hidden) |
|------|---------|-------|-------------|------------------|
| 2P FFA | 2 | None | 10 each | 35 |
| 3P FFA | 3 | None | 10 each | 25 |
| 4P FFA | 4 | None | 10 each | 15 |
| 4P Pairs | 4 | Seats 1&3 vs 2&4 | 10 each | 15 |

- Tiles are dealt randomly. Undealt tiles remain face-down and unknown to all players for the entire round.

### Turn Order & First Move
- The player holding the **highest double** opens the round (9-9 → 8-8 → 7-7 → ... → 0-0). That player may open with **any tile in their hand** (not necessarily the double).
- Play proceeds clockwise.
- In Pairs, partners sit across from each other (turn order alternates teams: Team A → Team B → Team A → Team B).

### Playing
- The board is a chain of tiles. Each end of the chain has an exposed number.
- On your turn, you must play a tile from your hand that matches **one** of the two open ends.
- The matching side connects to the chain; the other side becomes the new open end.
- If you cannot play, you **pass** (no drawing from undealt tiles).

### Winning
- **Empty hand**: First player to play all their tiles wins the round.
- **Stalemate (locked board)**: If all players pass consecutively (no one can play), the player with the **lowest sum of pip values** on remaining tiles wins. **Ties are possible.**

### Scoring
| Mode | Tracking |
|------|----------|
| FFA (2P, 3P, 4P) | Win Rate = Wins / (Wins + Losses), ties excluded |
| Pairs (4P) | Win Rate + **points scored** (sum of opposing team's remaining pip values) |

---

## 3. AI Strategies

### 3.1 Random (Baseline)
- Selects uniformly at random from all valid plays.
- Passes only when forced.

### 3.2 Greedy
- Plays the valid tile with the **highest pip sum** (e.g., 9-7 = 16 is preferred over 4-3 = 7).
- Tiebreaker: random among tied tiles.

### 3.3 Reinforcement Learning Agent
- Trained via self-play and play against other strategies.
- Learns from hidden-information game dynamics.
- See [Section 7: RL Design](#7-reinforcement-learning-design) for details.

---

## 4. Ranking & Metrics

### Primary: Win Rate
- **Win Rate (%) = Wins / (Wins + Losses)** — ties (stalemate draws) are excluded from the calculation.
- A "win" is a **team outcome**: if either teammate empties their hand or the team has lowest pip sum in a stalemate, both teammates get a win. Otherwise it's a loss.
- Ranked per game-mode (2P, 3P, 4P FFA, 4P Pairs).

### Secondary Metrics
| Metric | Description |
|--------|-------------|
| Avg. Remaining Pips | Average pip sum in hand at end of game (lower = better) |
| Points Scored (Pairs) | Average opponent pip sum when winning |

---

## 5. Architecture

### High-Level Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   GitHub Pages (Static)                  │
│  ┌─────────────────────┐  ┌───────────────────────────┐ │
│  │ Simulation Dashboard │  │    Interactive Game UI     │ │
│  │  (reads JSON data)   │  │ (JS engine + ONNX model)  │ │
│  └─────────────────────┘  └───────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │ static JSON + ONNX model
                          │
┌─────────────────────────────────────────────────────────┐
│              Local / CI Pipeline (Python)                 │
│  ┌──────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│  │  Game     │ │  Simulation  │ │   RL Training        │ │
│  │  Engine   │ │  Runner      │ │   Pipeline           │ │
│  └──────────┘ └──────────────┘ └──────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Backend (Python)
- **Game Engine**: Core dominos logic — tile management, board state, turn validation, win detection.
- **Strategies**: Pluggable strategy interface. Each strategy receives game state and returns an action.
- **Simulation Runner**: Runs N games across all strategy matchups and game modes. Outputs results as JSON (win rates, team stats, game records).
- **RL Training Pipeline**: Trains the RL agent, exports model weights.
- **Local Dev Server**: `python3 -m http.server` in `frontend/public/` for local testing.

### Frontend (Vanilla JS)
- **Self-contained static HTML/JS** — no build step, no framework, deployable to GitHub Pages as-is.
- **Game Engine (JS)**: Re-implementation of Python game logic for client-side play. Validated via `test-engine.html` test suite.
- **Model Inference**: Loads ONNX model via **ONNX Runtime Web** (CDN) for browser-side RL inference.

---

## 6. UI Design

### 6.1 Screen: Simulation Dashboard (Default View)

The landing page. Shows the latest simulation results. **Retro/pixel-art aesthetic** — pixel font (e.g., Press Start 2P or similar), dark CRT-inspired background, pixelated tile/domino graphics, scanline or noise effects, neon accent colors on dark backgrounds.

```
┌──────────────────────────────────────────────────────┐
│  ░░ CUBAN DOMINOS ░░                                 │
│  ░░ ML STRATEGY ARENA ░░              [PLAY ▶]       │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌─ WIN RATE LEADERBOARD ────────────────────────┐   │
│  │  Mode: [4P Pairs]                             │   │
│  │                                               │   │
│  │  #1  RL Agent    ████████████████░  72.3% WR  │   │
│  │  #2  Greedy      ███████████░░░░░  58.1% WR   │   │
│  │  #3  Random      ██████░░░░░░░░░░  34.2% WR   │   │
│  └───────────────────────────────────────────────┘   │
│                                                      │
│  ┌─ CHARTS ──────────────────────────────────────┐   │
│  │  • Win rate over time (line chart)            │   │
│  │  • Win rate bar chart                         │   │
│  │  • Avg remaining pips (bar chart)             │   │
│  │  • Pairs: points scored distribution          │   │
│  └───────────────────────────────────────────────┘   │
│                                                      │
│  ┌─ RECENT GAMES ────────────────────────────────┐   │
│  │  Game #10042  4P Pairs  Winner: RL Agent      │   │
│  │  Game #10041  4P Pairs  Winners: Greedy+RL    │   │
│  │  ...                                          │   │
│  └───────────────────────────────────────────────┘   │
│                                                      │
└──────────────────────────────────────────────────────┘
```

**Visual style notes:**
- Pixel/bitmap font for all text (Google Fonts: "Press Start 2P" or "VT323")
- Dark background (#0a0a0a or deep navy) with CRT-green or neon accents
- Pixelated domino tile icons in the header or as decorative elements
- Optional: subtle scanline overlay, CRT screen curvature effect via CSS
- Bars use block characters (█░) style, rendered as chunky pixel bars
- Buttons styled as retro arcade/console buttons

### 6.2 Screen: Test Setup (via "Test" button)

```
┌──────────────────────────────────────────────────────┐
│  ← Back                  Test Mode                   │
├──────────────────────────────────────────────────────┤
│                                                      │
│          How many players?                           │
│                                                      │
│     ┌─────┐   ┌─────┐   ┌───────────┐               │
│     │  2  │   │  3  │   │ 4 (Pairs) │               │
│     └─────┘   └─────┘   └───────────┘               │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 6.3 Screen: Game Play

```
┌──────────────────────────────────────────────────────┐
│  ← Back              4P Pairs                        │
├──────────────────────────────────────────────────────┤
│                                                      │
│                 [▼ Greedy]                            │
│               Partner (Team A)                       │
│                  ▓▓▓▓▓▓▓▓▓▓                          │
│                                                      │
│   [▼ Random]                      [▼ RL Agent]       │
│   Opponent                          Opponent         │
│   (Team B)                          (Team B)         │
│    ▓▓▓▓▓▓▓▓                          ▓▓▓▓▓▓▓▓       │
│                                                      │
│          ┌─────────────────────────┐                 │
│          │   Board: [5|4][4|9][9|2]│                 │
│          └─────────────────────────┘                 │
│                                                      │
│              YOUR TILES (Team A)                     │
│   ┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐              │
│   │ 2|7 ││ 5|5 ││ 3|8 ││ 0|6 ││ 1|9 │              │
│   └─────┘└─────┘└─────┘└─────┘└─────┘              │
│                                                      │
│           [Pass]           Turn: YOU                  │
│                                                      │
└──────────────────────────────────────────────────────┘
```

- **Strategy dropdowns** appear above each AI player's seat.
- Player's tiles are visible; opponents' tiles are face-down (▓).
- Clickable tiles highlight valid plays. Invalid clicks are ignored.
- Pass button is only enabled when no valid plays exist.
- Game-over overlay shows winner, remaining tiles, and points (Pairs).

---

## 7. Reinforcement Learning Design

### Algorithm: PPO (Proximal Policy Optimization)

Clipped surrogate objective with Generalized Advantage Estimation (GAE). Chosen over DQN/REINFORCE for better sample efficiency and training stability in this high-variance environment.

### Network: DominoNet

4-layer MLP with residual connections (~951k params):

```
Input (511-dim) → Linear(512) + LayerNorm + ReLU
  → Linear(512) + LayerNorm + ReLU + residual
  → Linear(512) + LayerNorm + ReLU + residual
  → Linear(256) + LayerNorm + ReLU
  → policy_head(256 → 111)   # masked softmax over actions
  → value_head(256 → 1)      # win probability estimate
```

### State Encoding (511 dims)

Mirrors what a human player naturally tracks during a game:

| Feature | Dims | Description |
|---------|------|-------------|
| Hand tiles | 55 | Binary: which tiles I hold |
| Played tiles | 55 | Binary: what's on the board |
| Board ends | 21 | One-hot per end (10+10) + empty flag |
| Opponent tile counts | 3 | How many tiles each opponent holds |
| Pass history | 4 | Did each player pass last turn |
| Game flags | 9 | Seat, teammate, first move |
| Per-seat played tiles | 220 | What each player has played (4 x 55) |
| Pass counts | 4 | Cumulative passes per player |
| Tile play order | 55 | When each tile was played |
| Opponent missing values | 30 | Pip values opponents lack (from passes) |
| Opponent threat | 3 | 1 - (missing_count / 10) per opponent |
| Pip value counts | 10 | Fraction of each pip 0-9 played |
| End scarcity | 2 | Fraction of end-matching tiles played |
| Per-seat pip profile | 40 | Per-seat pip value play counts (4 x 10) |

### Action Space
- **111 possible actions**: Play tile T on left (55) or right (55) end, or pass (1).
- Masked to only valid actions each turn.

### Reward Design
| Event | Reward |
|-------|--------|
| Win round | +1.0 |
| Lose round | -1.0 |
| Tie | 0.0 |
| Pip-reduction shaping | +0.1 * (initial_pips - final_pips) / initial_pips |

### Training Regime (v2 — Phased Curriculum)

| Phase | Episodes | Opponents | Self-Play |
|-------|----------|-----------|-----------|
| Curriculum | 200k | Graduated: weak → all 8 strategies | 0% |
| Hard opponents | 300k | Top 4 only (NeverPassed, PassTracker, GreedyDoubles, PartnerAware) | 40% |
| Self-play | 500k | Top 4 + frozen self | 70% |

**Total: 1M episodes**, batch size 64, PPO epochs 4, GAE lambda 0.95, gamma 0.99.

### Evaluation

Round-robin with homogeneous team pairs. Each strategy pair plays every other pair with seats flipped halfway for fairness. 500 games per matchup, 36 matchups = 18,000 games. Strategies ranked by **win rate** = wins / (wins + losses), ties excluded.

### Model Export
- Train in PyTorch → export to **ONNX** format → load in browser via **ONNX Runtime Web**.

---

## 8. Tech Stack

| Component | Technology |
|-----------|------------|
| Game Engine (training) | Python 3.12+ |
| RL Framework | PyTorch |
| Simulation Data | JSON files (committed to repo) |
| Pixel Font | Press Start 2P (Google Fonts) |
| Frontend | Vanilla JS/HTML (no framework, no build step) |
| Browser ML Inference | ONNX Runtime Web (CDN) |
| Deployment | GitHub Pages |
| CI/CD | GitHub Actions |

---

## 9. Project Structure

```
simple-dominos/
├── PRD.md                 # This file
├── CLAUDE.md              # Code conventions & architecture
├── README.md              # Project overview & quick start
├── STATS.md               # Game statistics & mathematical analysis
├── TODO.md                # Current work items & helpful tips
├── backend/
│   ├── engine/
│   │   ├── types.py       # GameMode, End, InvalidMoveError
│   │   ├── tile.py        # Tile, Board, create_full_set()
│   │   ├── player.py      # Player dataclass
│   │   └── game.py        # Game, Action, GameState, RoundResult
│   ├── strategies/        # 8 rule-based + 1 RL strategy
│   │   ├── base.py        # Strategy ABC
│   │   └── *.py           # Random, Greedy, GreedyDoubles, NonGreedy,
│   │                      #   LateGame, NeverPassed, PassTracker,
│   │                      #   PartnerAware, RLStrategy
│   ├── rl/
│   │   ├── encoding.py    # GameState → 511-dim tensor
│   │   ├── network.py     # DominoNet (4-layer MLP, residual)
│   │   ├── trainer.py     # PPO + phased curriculum training
│   │   └── export.py      # PyTorch → ONNX
│   ├── simulation/
│   │   ├── runner.py      # SimulationRunner + RoundRobinRunner
│   │   └── results.py     # SimulationResults, JSON export
│   └── tests/             # 127 pytest tests
├── frontend/public/       # Static HTML/JS, GitHub Pages ready
│   ├── index.html         # Strategy battle viewer (animated leaderboard)
│   ├── play.html          # Interactive 4P game UI (human vs AI)
│   ├── test-engine.html   # JS engine test suite
│   ├── js/                # engine.js, strategies.js, encoding.js,
│   │                      #   rl-strategy.js, prng.js
│   ├── models/            # ONNX model for browser RL inference
│   └── data/              # Pre-computed simulation JSON
├── scripts/
│   ├── run_simulation.py  # CLI: batch simulation → JSON
│   ├── train_rl.py        # CLI: RL training (PPO, phased)
│   └── eval_rl.py         # CLI: round-robin evaluation
└── models/                # PyTorch checkpoints + ONNX exports
```

---

## 10. Milestones

### M1: Game Engine + Strategies ✅
- [x] Tile, board, and player data structures (Python)
- [x] Full game loop with all rules (Double-9, pass, stalemate, scoring)
- [x] Support for 2P, 3P, 4P Pairs (no 4P FFA)
- [x] 8 strategies: Random, Greedy, GreedyDoubles, NonGreedy, LateGame, NeverPassed, PassTracker, PartnerAware
- [x] Unit tests with full coverage (96 tests)

### M2: Simulation Runner ✅
- [x] Simulation runner (batch games across all strategy matchups)
- [x] Win rate tracking (wins / (wins + losses), ties excluded)
- [x] JSON output of simulation results
- [x] CLI entry point: `scripts/run_simulation.py`
- [x] Test suite (24 tests, 120 total)

### M3: Strategy Battle Viewer ✅
- [x] Single-file HTML dashboard at `frontend/public/index.html`
- [x] Animated leaderboard — click Start, watch strategies race by win rate
- [x] 4P Pairs only (random matchups, 100k games)
- [x] Game browser with search/filter (strategy, result, stalemate)
- [x] Team composition leaderboard
- [x] Deployable on GitHub Pages (no build step, no server)
- [x] **Retro/pixel-art redesign** — Press Start 2P font, CRT aesthetic, scanlines
- [x] **Remove Elo** — rank by win rate only (wins/(wins+losses), ties excluded)

### M4: Frontend — Interactive Game ✅
- [x] JavaScript game engine (vanilla JS, mirrors Python engine)
- [x] Game board with SVG tile rendering (`play.html`)
- [x] Strategy selection dropdowns per AI player (all 8 rule-based + RL)
- [x] Turn-by-turn gameplay with pass support
- [x] Game-over summary
- [x] Engine test suite (`test-engine.html`)

### M5: Reinforcement Learning ✅
- [x] State encoding (`rl/encoding.py` — 511-dim feature vector, human-style tile tracking)
- [x] DominoNet architecture (`rl/network.py` — 4-layer MLP with residual connections)
- [x] PPO training loop with GAE (`rl/trainer.py`)
- [x] Phased curriculum trainer (3 phases, graduated opponents, self-play ramp)
- [x] PyTorch → ONNX export (`rl/export.py`)
- [x] RL strategy wrapper (`strategies/rl_strategy.py`)
- [x] CLI entry point (`scripts/train_rl.py`)
- [x] Round-robin evaluation (`RoundRobinRunner` + head-to-head matrix)
- [x] v1 PPO training (500k episodes) — #1 ranked, 57.3% win rate
- [x] v2 training (1M episodes, phased curriculum) — #1 ranked, 57.2% WR, beats NeverPassed h2h
- [x] Evaluation tournament (2000 games/matchup, 72k total)
- [x] Validated ONNX model ready for browser (3.7MB embedded)

### M6: Full Integration — In Progress
- [x] RL model loaded in browser via ONNX Runtime Web
- [x] RL strategy playable in `play.html` interactive mode
- [x] Updated simulation with RL strategy included in win rate rankings
- [ ] CI/CD: train → simulate → export → deploy
- [ ] Final leaderboard with all strategies ranked by win rate

---

## 11. Open Questions

| # | Question | Status |
|---|----------|--------|
| 1 | TypeScript re-implementation vs Pyodide for browser game engine? | **Decided M4**: vanilla JS re-implementation (no build step) |
| 2 | DQN vs PPO — evaluate after initial training results | Decided M5: PPO (much better than REINFORCE/DQN) |
| 3 | Should RL agent have separate models per game mode (2P, 3P, 4P)? | Deferred — 4P Pairs only for now |
| 4 | Add more strategies later? (e.g., MCTS with information set sampling) | Future scope |
