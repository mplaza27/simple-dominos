#!/usr/bin/env python3
"""CLI entry point for running dominos strategy simulations."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Add backend/ to path so imports work from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from simulation.runner import SimulationRunner
from strategies.greedy_doubles_strategy import GreedyDoublesStrategy
from strategies.greedy_strategy import GreedyStrategy
from strategies.late_game_strategy import LateGameStrategy
from strategies.never_passed_strategy import NeverPassedStrategy
from strategies.non_greedy_strategy import NonGreedyStrategy
from strategies.partner_aware_strategy import PartnerAwareStrategy
from strategies.pass_tracker_strategy import PassTrackerStrategy
from strategies.random_strategy import RandomStrategy
from strategies.rl_strategy import RLStrategy

DEFAULT_RL_MODEL = Path(__file__).resolve().parent.parent / "models" / "domino_rl_v2.pt"


def make_strategies(rng: random.Random, include_rl: bool = True) -> list:
    strategies = [
        RandomStrategy(rng=random.Random(rng.randint(0, 2**63))),
        GreedyStrategy(rng=random.Random(rng.randint(0, 2**63))),
        GreedyDoublesStrategy(rng=random.Random(rng.randint(0, 2**63))),
        NonGreedyStrategy(rng=random.Random(rng.randint(0, 2**63))),
        LateGameStrategy(rng=random.Random(rng.randint(0, 2**63))),
        NeverPassedStrategy(rng=random.Random(rng.randint(0, 2**63))),
        PassTrackerStrategy(rng=random.Random(rng.randint(0, 2**63))),
        PartnerAwareStrategy(rng=random.Random(rng.randint(0, 2**63))),
    ]
    if include_rl and DEFAULT_RL_MODEL.exists():
        strategies.append(RLStrategy(model_path=DEFAULT_RL_MODEL, hidden_dim=512))
    return strategies


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dominos strategy simulations")
    parser.add_argument("--games", type=int, default=100000, help="Number of games (default: 100000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--no-rl", action="store_true", help="Exclude RL strategy")
    parser.add_argument(
        "--output",
        type=str,
        default="frontend/public/data/results.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    strategies = make_strategies(rng, include_rl=not args.no_rl)

    runner = SimulationRunner(
        strategies=strategies,
        num_games=args.games,
        rng=random.Random(rng.randint(0, 2**63)),
    )

    def progress(done: int, total: int) -> None:
        pct = done * 100 // total
        print(f"\r  Progress: {done}/{total} games ({pct}%)", end="", flush=True)

    runner.set_progress_callback(progress)

    print(f"Running simulation: {args.games} games (4P Pairs), seed={args.seed}")
    print(f"Strategies: {[s.name for s in strategies]}")
    print()

    results = runner.run()
    print()  # newline after progress

    # Print strategy leaderboard
    print(f"\n{'=' * 60}")
    print(f"  Strategy Leaderboard")
    print(f"{'=' * 60}")
    print(f"  {'Strategy':<25} {'Elo':>7} {'Win%':>7} {'W':>5} {'L':>5} {'D':>5} {'GP':>6}")
    print(f"  {'-' * 61}")
    for s in results.leaderboard:
        print(
            f"  {s.name:<25} {s.elo:>7.1f} {s.win_rate:>6.1%} {s.wins:>5} "
            f"{s.losses:>5} {s.draws:>5} {s.games_played:>6}"
        )

    # Print top team compositions
    print(f"\n{'=' * 60}")
    print(f"  Top Team Compositions (min 50 games)")
    print(f"{'=' * 60}")
    print(f"  {'Team':<35} {'Win%':>7} {'W':>5} {'L':>5} {'D':>5} {'GP':>6}")
    print(f"  {'-' * 64}")
    top_teams = [t for t in results.team_leaderboard if t.games_played >= 50]
    for t in top_teams[:15]:
        team_str = f"{t.team[0]} + {t.team[1]}"
        print(
            f"  {team_str:<35} {t.win_rate:>6.1%} {t.wins:>5} "
            f"{t.losses:>5} {t.draws:>5} {t.games_played:>6}"
        )

    # Save JSON
    output_path = Path(args.output)
    results.save(output_path)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
