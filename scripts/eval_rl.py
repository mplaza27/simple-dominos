#!/usr/bin/env python3
"""Evaluate the trained RL agent against all rule-based strategies.

Uses round-robin matchups: each pair of strategies plays as homogeneous
teams (Team A vs Team B), with seat assignment flipped halfway to control
for first-player advantage.

Usage:
    python scripts/eval_rl.py --model models/domino_rl.pt --games-per-matchup 200
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from simulation.runner import RoundRobinRunner
from strategies.greedy_doubles_strategy import GreedyDoublesStrategy
from strategies.greedy_strategy import GreedyStrategy
from strategies.late_game_strategy import LateGameStrategy
from strategies.never_passed_strategy import NeverPassedStrategy
from strategies.non_greedy_strategy import NonGreedyStrategy
from strategies.partner_aware_strategy import PartnerAwareStrategy
from strategies.pass_tracker_strategy import PassTrackerStrategy
from strategies.random_strategy import RandomStrategy
from strategies.rl_strategy import RLStrategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RL agent vs all strategies")
    parser.add_argument("--model", type=str, default="models/domino_rl.pt",
                        help="Path to trained model (default: models/domino_rl.pt)")
    parser.add_argument("--games-per-matchup", type=int, default=200,
                        help="Games per matchup pair (default: 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="If set, save results JSON to this path")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    strategies = [
        RandomStrategy(rng=random.Random(rng.randint(0, 2**63))),
        GreedyStrategy(rng=random.Random(rng.randint(0, 2**63))),
        GreedyDoublesStrategy(rng=random.Random(rng.randint(0, 2**63))),
        NonGreedyStrategy(rng=random.Random(rng.randint(0, 2**63))),
        LateGameStrategy(rng=random.Random(rng.randint(0, 2**63))),
        NeverPassedStrategy(rng=random.Random(rng.randint(0, 2**63))),
        PassTrackerStrategy(rng=random.Random(rng.randint(0, 2**63))),
        PartnerAwareStrategy(rng=random.Random(rng.randint(0, 2**63))),
        RLStrategy(model_path=args.model, hidden_dim=512, device="cpu"),
    ]

    n_strats = len(strategies)
    n_matchups = n_strats * (n_strats - 1) // 2
    total_games = n_matchups * args.games_per_matchup

    runner = RoundRobinRunner(
        strategies=strategies,
        games_per_matchup=args.games_per_matchup,
        rng=random.Random(rng.randint(0, 2**63)),
    )

    def progress(done: int, total: int) -> None:
        pct = done * 100 // total
        print(f"\r  Progress: {done}/{total} games ({pct}%)", end="", flush=True)

    runner.set_progress_callback(progress)

    print(f"RL Evaluation: {n_matchups} matchups x {args.games_per_matchup} = {total_games} games (4P Pairs, round-robin), seed={args.seed}")
    print(f"Model: {args.model}")
    print(f"Strategies: {[s.name for s in strategies]}")
    print()

    results = runner.run()
    print()

    # -- Leaderboard --
    print(f"\n{'=' * 60}")
    print(f"  Strategy Leaderboard (with RL)")
    print(f"{'=' * 60}")
    print(f"  {'Strategy':<25} {'Elo':>7} {'Win%':>7} {'W':>5} {'L':>5} {'D':>5} {'GP':>6}")
    print(f"  {'-' * 61}")
    for s in results.leaderboard:
        marker = " <-- RL" if s.name == "RLStrategy" else ""
        print(
            f"  {s.name:<25} {s.elo:>7.1f} {s.win_rate:>6.1%} {s.wins:>5} "
            f"{s.losses:>5} {s.draws:>5} {s.games_played:>6}{marker}"
        )

    # -- Head-to-head summary table --
    # Build a lookup: (strat_a_name, strat_b_name) -> {wins_a, wins_b, draws}
    h2h: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"wins_a": 0, "wins_b": 0, "draws": 0})
    for g in results.games:
        # team_a and team_b are sorted tuples of strategy names (homogeneous, so both entries same)
        ta_name = g.team_a[0]  # homogeneous team
        tb_name = g.team_b[0]
        # Canonical key: alphabetical order
        key = tuple(sorted([ta_name, tb_name]))
        rec = h2h[key]
        if g.winner_team is None:
            rec["draws"] += 1
        elif g.winner_team == "A":
            # team_a won; figure out which canonical side that is
            if ta_name == key[0]:
                rec["wins_a"] += 1
            else:
                rec["wins_b"] += 1
        else:
            if tb_name == key[0]:
                rec["wins_a"] += 1
            else:
                rec["wins_b"] += 1

    strat_names = sorted([s.name for s in strategies])

    print(f"\n{'=' * 60}")
    print(f"  Head-to-Head Win Rates")
    print(f"{'=' * 60}")

    # Column width for strategy names
    cw = 8  # abbreviated column width
    header_names = [n[:cw] for n in strat_names]
    print(f"  {'':>25}", end="")
    for h in header_names:
        print(f" {h:>{cw}}", end="")
    print()
    print(f"  {'-' * (25 + (cw + 1) * len(strat_names))}")

    for row_name in strat_names:
        print(f"  {row_name:<25}", end="")
        for col_name in strat_names:
            if row_name == col_name:
                print(f" {'---':>{cw}}", end="")
            else:
                key = tuple(sorted([row_name, col_name]))
                rec = h2h[key]
                total = rec["wins_a"] + rec["wins_b"] + rec["draws"]
                if total == 0:
                    print(f" {'n/a':>{cw}}", end="")
                else:
                    # wins for row_name
                    if row_name == key[0]:
                        row_wins = rec["wins_a"]
                    else:
                        row_wins = rec["wins_b"]
                    wr = row_wins / total
                    print(f" {wr:>{cw}.1%}", end="")
        print()

    if args.output:
        output_path = Path(args.output)
        results.save(output_path)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
