from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pytest

from simulation.elo import EloSystem
from simulation.results import EloSnapshot, GameRecord, SimulationResults, StrategyStats, TeamStats
from simulation.runner import SimulationRunner
from strategies.greedy_strategy import GreedyStrategy
from strategies.random_strategy import RandomStrategy


# ── Elo tests ──────────────────────────────────────────────────────


class TestElo:
    def test_initial_rating(self) -> None:
        elo = EloSystem()
        assert elo.get_rating("Alice") == 1500.0

    def test_winner_gains_loser_loses(self) -> None:
        elo = EloSystem()
        elo.update("Alice", "Bob")
        assert elo.get_rating("Alice") > 1500.0
        assert elo.get_rating("Bob") < 1500.0

    def test_ratings_are_zero_sum(self) -> None:
        elo = EloSystem()
        elo.update("Alice", "Bob")
        total = elo.get_rating("Alice") + elo.get_rating("Bob")
        assert abs(total - 3000.0) < 0.01

    def test_k_factor_decay(self) -> None:
        elo = EloSystem(decay_games=10)
        for _ in range(10):
            elo.update("Alice", "Bob")
        k = elo._k_factor("Alice")
        assert k == 16.0

    def test_k_factor_initial(self) -> None:
        elo = EloSystem()
        assert elo._k_factor("NewPlayer") == 32.0

    def test_draw_handling(self) -> None:
        elo = EloSystem()
        elo.update("Alice", "Bob", draw=True)
        assert abs(elo.get_rating("Alice") - 1500.0) < 0.01
        assert abs(elo.get_rating("Bob") - 1500.0) < 0.01

    def test_draw_with_unequal_ratings(self) -> None:
        elo = EloSystem()
        for _ in range(5):
            elo.update("Alice", "Bob")
        alice_before = elo.get_rating("Alice")
        bob_before = elo.get_rating("Bob")
        elo.update("Alice", "Bob", draw=True)
        assert elo.get_rating("Alice") < alice_before
        assert elo.get_rating("Bob") > bob_before

    def test_multiplayer_pairwise(self) -> None:
        elo = EloSystem()
        elo.update_multiplayer(winners=["Alice"], losers=["Bob", "Charlie"])
        assert elo.get_rating("Alice") > 1500.0
        assert elo.get_rating("Bob") < 1500.0
        assert elo.get_rating("Charlie") < 1500.0

    def test_multiplayer_tied_winners(self) -> None:
        elo = EloSystem()
        elo.update_multiplayer(winners=["Alice", "Bob"], losers=["Charlie"])
        assert elo.get_rating("Alice") > 1500.0
        assert elo.get_rating("Bob") > 1500.0
        assert elo.get_rating("Charlie") < 1500.0

    def test_get_all_ratings(self) -> None:
        elo = EloSystem()
        elo.update("Alice", "Bob")
        ratings = elo.get_all_ratings()
        assert "Alice" in ratings
        assert "Bob" in ratings
        assert len(ratings) == 2


# ── Runner tests ───────────────────────────────────────────────────


class TestRunner:
    def test_exact_game_count(self) -> None:
        rng = random.Random(42)
        strategies = [
            RandomStrategy(rng=random.Random(1)),
            GreedyStrategy(rng=random.Random(2)),
        ]
        runner = SimulationRunner(
            strategies=strategies,
            num_games=100,
            rng=rng,
        )
        results = runner.run()
        assert len(results.games) == 100

    def test_all_strategies_appear(self) -> None:
        rng = random.Random(42)
        strategies = [
            RandomStrategy(rng=random.Random(1)),
            GreedyStrategy(rng=random.Random(2)),
        ]
        runner = SimulationRunner(
            strategies=strategies,
            num_games=200,
            rng=rng,
        )
        results = runner.run()
        seen = set()
        for g in results.games:
            seen.update(g.strategies)
        assert "RandomStrategy" in seen
        assert "GreedyStrategy" in seen

    def test_team_compositions_vary(self) -> None:
        rng = random.Random(42)
        strategies = [
            RandomStrategy(rng=random.Random(1)),
            GreedyStrategy(rng=random.Random(2)),
        ]
        runner = SimulationRunner(
            strategies=strategies,
            num_games=200,
            rng=rng,
        )
        results = runner.run()
        team_as = {g.team_a for g in results.games}
        # With 2 strategies, possible teams: (G,G), (R,R), (G,R)
        assert len(team_as) > 1

    def test_game_record_fields(self) -> None:
        rng = random.Random(42)
        strategies = [
            RandomStrategy(rng=random.Random(1)),
            GreedyStrategy(rng=random.Random(2)),
        ]
        runner = SimulationRunner(
            strategies=strategies,
            num_games=10,
            rng=rng,
        )
        results = runner.run()
        g = results.games[0]
        assert len(g.strategies) == 4
        assert len(g.team_a) == 2
        assert len(g.team_b) == 2
        assert g.winner_team in ("A", "B", None)
        assert isinstance(g.is_stalemate, bool)
        assert isinstance(g.team_a_pips, int)
        assert isinstance(g.team_b_pips, int)
        assert isinstance(g.points, int)

    def test_leaderboard_is_flat_list(self) -> None:
        rng = random.Random(42)
        strategies = [
            RandomStrategy(rng=random.Random(1)),
            GreedyStrategy(rng=random.Random(2)),
        ]
        runner = SimulationRunner(
            strategies=strategies,
            num_games=50,
            rng=rng,
        )
        results = runner.run()
        assert isinstance(results.leaderboard, list)
        assert len(results.leaderboard) == 2
        names = {s.name for s in results.leaderboard}
        assert names == {"RandomStrategy", "GreedyStrategy"}

    def test_team_leaderboard_present(self) -> None:
        rng = random.Random(42)
        strategies = [
            RandomStrategy(rng=random.Random(1)),
            GreedyStrategy(rng=random.Random(2)),
        ]
        runner = SimulationRunner(
            strategies=strategies,
            num_games=50,
            rng=rng,
        )
        results = runner.run()
        assert len(results.team_leaderboard) > 0
        for ts in results.team_leaderboard:
            assert ts.games_played > 0
            assert ts.wins + ts.losses + ts.draws == ts.games_played

    def test_progress_callback(self) -> None:
        rng = random.Random(42)
        strategies = [RandomStrategy(rng=random.Random(1))]
        runner = SimulationRunner(
            strategies=strategies,
            num_games=20,
            rng=rng,
        )
        calls: list[tuple[int, int]] = []
        runner.set_progress_callback(lambda done, total: calls.append((done, total)))
        runner.run()
        assert len(calls) == 20
        assert calls[-1] == (20, 20)


# ── Results / JSON tests ──────────────────────────────────────────


class TestResults:
    def test_json_serialization(self) -> None:
        results = SimulationResults(
            leaderboard=[
                StrategyStats(
                    name="GreedyStrategy",
                    elo=1520.0,
                    wins=7,
                    losses=3,
                    draws=0,
                    games_played=10,
                    avg_remaining_pips=5.5,
                    points_scored_avg=12.0,
                )
            ],
            team_leaderboard=[
                TeamStats(
                    team=("GreedyStrategy", "RandomStrategy"),
                    wins=5,
                    losses=3,
                    draws=2,
                    games_played=10,
                )
            ],
            elo_history={
                "GreedyStrategy": [EloSnapshot(game_number=1, elo=1516.0)]
            },
            games=[
                GameRecord(
                    game_id=0,
                    strategies=["GreedyStrategy", "RandomStrategy", "GreedyStrategy", "RandomStrategy"],
                    team_a=("GreedyStrategy", "GreedyStrategy"),
                    team_b=("RandomStrategy", "RandomStrategy"),
                    winner_team="A",
                    is_stalemate=False,
                    pip_sums={0: 0, 1: 25, 2: 0, 3: 18},
                    team_a_pips=0,
                    team_b_pips=43,
                    points=43,
                    moves=[[0, 5, 3, "l"], [1], [2, 7, 2, "r"]],
                )
            ],
        )
        data = results.to_json()
        assert "leaderboard" in data
        assert "team_leaderboard" in data
        assert "elo_history" in data
        assert "games" in data
        assert "total_games" in data
        assert data["total_games"] == 1

        # Should be JSON-serializable
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["leaderboard"][0]["name"] == "GreedyStrategy"
        assert parsed["games"][0]["team_a"] == ["GreedyStrategy", "GreedyStrategy"]
        assert parsed["games"][0]["winner_team"] == "A"
        assert parsed["team_leaderboard"][0]["team"] == ["GreedyStrategy", "RandomStrategy"]

    def test_json_required_fields(self) -> None:
        results = SimulationResults()
        data = results.to_json()
        assert "leaderboard" in data
        assert "team_leaderboard" in data
        assert "elo_history" in data
        assert "games" in data
        assert "total_games" in data

    def test_save_to_file(self) -> None:
        results = SimulationResults(
            leaderboard=[],
            team_leaderboard=[],
            elo_history={},
            games=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "results.json"
            results.save(path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["total_games"] == 0

    def test_win_rate_property(self) -> None:
        s = StrategyStats(
            name="Test", elo=1500, wins=3, losses=7, draws=0,
            games_played=10, avg_remaining_pips=0, points_scored_avg=0,
        )
        assert abs(s.win_rate - 0.3) < 0.001

    def test_win_rate_zero_games(self) -> None:
        s = StrategyStats(
            name="Test", elo=1500, wins=0, losses=0, draws=0,
            games_played=0, avg_remaining_pips=0, points_scored_avg=0,
        )
        assert s.win_rate == 0.0

    def test_team_stats_win_rate(self) -> None:
        ts = TeamStats(
            team=("A", "B"), wins=7, losses=2, draws=1, games_played=10,
        )
        assert abs(ts.win_rate - 0.7) < 0.001

    def test_elo_history_length(self) -> None:
        rng = random.Random(42)
        strategies = [
            RandomStrategy(rng=random.Random(1)),
            GreedyStrategy(rng=random.Random(2)),
        ]
        runner = SimulationRunner(
            strategies=strategies,
            num_games=50,
            rng=rng,
        )
        results = runner.run()
        total_games = len(results.games)
        for name, snaps in results.elo_history.items():
            assert len(snaps) > 0
            assert len(snaps) <= total_games
            for i in range(1, len(snaps)):
                assert snaps[i].game_number >= snaps[i - 1].game_number


# ── Smoke test ────────────────────────────────────────────────────


class TestSmoke:
    def test_full_run_all_strategies(self) -> None:
        """All 8 strategies x 500 games completes without error."""
        from strategies.greedy_doubles_strategy import GreedyDoublesStrategy
        from strategies.late_game_strategy import LateGameStrategy
        from strategies.never_passed_strategy import NeverPassedStrategy
        from strategies.non_greedy_strategy import NonGreedyStrategy
        from strategies.partner_aware_strategy import PartnerAwareStrategy
        from strategies.pass_tracker_strategy import PassTrackerStrategy

        rng = random.Random(42)
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
        runner = SimulationRunner(
            strategies=strategies,
            num_games=500,
            rng=random.Random(rng.randint(0, 2**63)),
        )
        results = runner.run()

        assert len(results.games) == 500
        assert len(results.leaderboard) == 8
        assert len(results.team_leaderboard) > 0


# ── Reproducibility test ──────────────────────────────────────────


class TestReproducibility:
    def test_same_seed_same_results(self) -> None:
        def run_with_seed(seed: int) -> dict[str, float]:
            rng = random.Random(seed)
            strategies = [
                RandomStrategy(rng=random.Random(rng.randint(0, 2**63))),
                GreedyStrategy(rng=random.Random(rng.randint(0, 2**63))),
            ]
            runner = SimulationRunner(
                strategies=strategies,
                num_games=50,
                rng=random.Random(rng.randint(0, 2**63)),
            )
            results = runner.run()
            return {s.name: s.elo for s in results.leaderboard}

        r1 = run_with_seed(42)
        r2 = run_with_seed(42)
        for name in r1:
            assert abs(r1[name] - r2[name]) < 0.001, f"{name}: {r1[name]} != {r2[name]}"

    def test_different_seed_different_results(self) -> None:
        def run_with_seed(seed: int) -> dict[str, float]:
            rng = random.Random(seed)
            strategies = [
                RandomStrategy(rng=random.Random(rng.randint(0, 2**63))),
                GreedyStrategy(rng=random.Random(rng.randint(0, 2**63))),
            ]
            runner = SimulationRunner(
                strategies=strategies,
                num_games=50,
                rng=random.Random(rng.randint(0, 2**63)),
            )
            results = runner.run()
            return {s.name: s.elo for s in results.leaderboard}

        r1 = run_with_seed(42)
        r2 = run_with_seed(99)
        any_different = any(abs(r1[n] - r2[n]) > 0.01 for n in r1)
        assert any_different
