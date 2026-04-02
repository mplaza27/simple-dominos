# Simple Dominos — Cuban Dominos (Double-9) Game

## Source of Truth
- **PRD.md** is the authoritative source for project scope, milestones, and next steps
- **TODO.md** tracks immediate work items (always points back to PRD milestones)
- This file (CLAUDE.md) covers code conventions and architecture only

## Game Rules
- Double-9 domino set (55 tiles, pips 0-9)
- 4P Pairs only: Team A = seats 0,2; Team B = seats 1,3
- Each player is dealt 10 tiles; remaining 15 tiles are out of play (hidden)
- First player determined by: highest double in hand; if no doubles, highest pip-sum tile
- First player can play any tile (not restricted to their qualifying tile)
- Play alternates clockwise; must match an open end or pass
- Round ends when a player empties their hand (win) or all players pass consecutively (stalemate)
- Stalemate winner: individual player with lowest pip sum wins for their team; ties if both teams have a player at the minimum
- Points = opposing team's remaining pips

## Tech Stack
- **Backend**: Python 3.12+, pure stdlib for engine (no external deps except pytest)
- **Frontend**: Vanilla JS/HTML (self-contained, no build step, GitHub Pages ready)
- **RL Training**: PyTorch
- **Browser Inference**: ONNX Runtime Web (onnxruntime-web CDN)
- **Ranking**: Win Rate = Wins / (Wins + Losses), ties excluded

## Code Conventions
- Type hints everywhere
- Frozen dataclasses for immutable data (`slots=True` for performance)
- `rng` injection (pass `random.Random` instances) for reproducibility
- pytest for all testing
- No circular imports: use `TYPE_CHECKING` guards
- Define shared types in `engine/types.py`

## Architecture
```
backend/
├── engine/          # Core game logic (types, tile, player, game)
├── strategies/      # Pluggable AI strategies (base ABC, 8 rule-based + 1 RL)
├── rl/              # RL training pipeline (encoding, network, trainer, export)
├── simulation/      # Batch runner, win rate tracking, results/JSON export
└── tests/           # pytest test suite (127 tests)
frontend/
└── public/
    ├── index.html   # Strategy battle viewer (animated leaderboard)
    ├── play.html    # Interactive 4P game UI (human vs AI)
    ├── test-engine.html  # JS engine test suite
    ├── js/          # engine.js, strategies.js, encoding.js, rl-strategy.js, prng.js
    ├── models/      # ONNX model for browser RL inference
    └── data/
        └── results.json  # Pre-computed simulation data
models/              # PyTorch checkpoints + ONNX exports
scripts/
├── run_simulation.py    # CLI: run sims, output JSON
├── train_rl.py          # CLI: RL training (PPO, phased curriculum)
└── eval_rl.py           # CLI: round-robin evaluation
```

- Import pattern: `from engine.tile import Tile, Board` (backend/ is Python root)
- Hidden info rule: GameState must NEVER expose other players' hands or undealt tiles
- Strategy interface: `Strategy` ABC in `strategies/base.py`
- Core types: `GameState`/`Action` in `engine/game.py`

## Dependency Order
`types.py` -> `tile.py` -> `player.py` -> `game.py` -> `strategies/` -> `tests/`
