# TODO — Simple Dominos

> Source of truth: **PRD.md**. Items here map to PRD milestones.

## Current (M6: Full Integration)
- [x] Load ONNX model in `play.html` via ONNX Runtime Web (onnxruntime-web 1.21.0 CDN)
- [x] RL selectable in strategy dropdowns (RLStrategy in STRATEGY_REGISTRY)
- [x] Async inference via `chooseActionAsync()` with greedy fallback
- [x] Added RL to `run_simulation.py` (`--no-rl` flag to exclude)
- [x] Full 100k simulation with RL → `results.json` updated (RL+RL = #1 team at 57.5% WR)
- [ ] CI/CD pipeline: train → simulate → export → deploy
- [ ] Final leaderboard with all strategies ranked by win rate

## Completed Milestones

### M5: Reinforcement Learning ✅
- v2 training (1M episodes, phased curriculum, `models/domino_rl_v2.pt`)
- 4-layer residual network (951k params), STATE_DIM=511
- Phase 1: curriculum (200k), Phase 2: hard opponents + 40% self-play (300k), Phase 3: 70% self-play (500k)
- Evaluated v2 (72k games, 2000/matchup): #1 ranked, 57.2% WR, beats NeverPassed h2h (51.4%)
- Exported ONNX model (3.7MB embedded) — validated against PyTorch (max diff <1e-6)

### v2 Results (72k games, 2000/matchup)

Win Rate = Wins / (Wins + Losses), ties excluded.

| Metric | v1 | v2 actual |
|--------|-----|-----------|
| Overall rank | #1 (57.3% WR) | #1 (57.2% WR) |
| vs NeverPassed h2h | 43.5% | 51.4% ✓ |
| vs top-4 avg h2h | ~52% | ~51.3% |
| vs Random h2h | 64% | 60.9% |

Note: 60%+ overall WR may be unrealistic given game's inherent variance (15 hidden tiles). v2 improved key matchup vs NeverPassed.

### M1–M4 ✅
- Game engine, 8 rule-based strategies, simulation runner, strategy battle viewer, interactive game UI — all complete. See PRD.md for details.

## Known Issues
- Training is CPU-bottlenecked by Python game simulation, not GPU compute
- Cuban Dominos 4P Pairs is inherently high-variance (15 hidden tiles, low branching factor)
  - See STATS.md for full mathematical analysis
  - Round-robin eval gives cleaner signal than random team pairing
- Old model checkpoints (state_dim=426 or 459) are incompatible with v2 encoding (state_dim=511)
  - `RLStrategy` handles backward compat via `state_dim` field in checkpoint

---

## Helpful Tips

### Setting up from scratch

**Prerequisites:** Python 3.12+, Ubuntu 24.04 (or similar). GPU optional (NVIDIA RTX 3080+, CUDA 13.0).

```bash
# Create venv
cd ~/code-projects/simple-dominos
python3 -m venv .venv
source .venv/bin/activate

# Core (game engine + tests)
pip install pytest

# RL training (pick one)
pip install torch --index-url https://download.pytorch.org/whl/cu128   # with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU only

# Verify
cd backend && ../.venv/bin/pytest tests/ -v   # 127 tests should pass
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Quick commands

```bash
# Run tests
python -m pytest backend/tests/ -v

# Simulation (100k games)
python scripts/run_simulation.py --games 100000 --seed 42

# Train RL (v2 phased, 1M episodes)
python scripts/train_rl.py --phased --save-path models/domino_rl_v2.pt

# Evaluate RL
python scripts/eval_rl.py --model models/domino_rl_v2.pt --games-per-matchup 500

# Frontend
cd frontend/public && python3 -m http.server 8080
# http://localhost:8080/play.html       — playable game
# http://localhost:8080/index.html      — strategy battle viewer
```

### RL Training Operations

```bash
# Check if training is running
ps aux | grep train_rl | grep -v grep

# Check training progress
python -c "import torch; c=torch.load('models/domino_rl_v2.pt', weights_only=False); print(f'Episode: {c[\"episode\"]}')"

# Resume v2 training from last checkpoint
python scripts/train_rl.py --phased --save-path models/domino_rl_v2.pt --resume
```

### Package summary

| Package | Purpose |
|---------|---------|
| `pytest` | Test runner |
| `torch` | RL training + inference |

That's it — the game engine is pure stdlib Python.
