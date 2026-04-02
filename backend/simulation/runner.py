from __future__ import annotations

import itertools
import random
from collections import defaultdict
from typing import Callable

from engine.game import Game
from engine.player import Player
from engine.types import GameMode
from simulation.elo import EloSystem
from simulation.results import (
    EloSnapshot,
    GameRecord,
    SimulationResults,
    StrategyStats,
    TeamStats,
)
from strategies.base import Strategy


class SimulationRunner:
    def __init__(
        self,
        strategies: list[Strategy],
        num_games: int,
        rng: random.Random,
    ) -> None:
        self._strategies = strategies
        self._num_games = num_games
        self._rng = rng
        self._elo = EloSystem()
        self._progress_cb: Callable[[int, int], None] | None = None

    def set_progress_callback(self, cb: Callable[[int, int], None]) -> None:
        self._progress_cb = cb

    def run(self) -> SimulationResults:
        results = SimulationResults()

        # Per-strategy accumulators
        stats: dict[str, _Accum] = defaultdict(_Accum)

        # Per-team-composition accumulators
        team_stats: dict[tuple[str, str], _TeamAccum] = defaultdict(_TeamAccum)

        # Thin Elo snapshots: keep ~500 evenly spaced + final
        elo_snap_interval = max(1, self._num_games // 500)

        for game_id in range(self._num_games):
            # Pick 4 strategies independently at random
            seat_strategies = [self._rng.choice(self._strategies) for _ in range(4)]

            record = self._run_one_game(game_id, seat_strategies, stats, team_stats)
            results.games.append(record)

            # Snapshot Elo at intervals
            game_num = game_id + 1
            if game_num % elo_snap_interval == 0 or game_num == self._num_games:
                for name in self._elo.get_all_ratings():
                    if name not in results.elo_history:
                        results.elo_history[name] = []
                    results.elo_history[name].append(
                        EloSnapshot(game_number=game_num, elo=self._elo.get_rating(name))
                    )

            if self._progress_cb:
                self._progress_cb(game_num, self._num_games)

        # Build strategy leaderboard
        for strat in self._strategies:
            acc = stats[strat.name]
            results.leaderboard.append(
                StrategyStats(
                    name=strat.name,
                    elo=self._elo.get_rating(strat.name),
                    wins=acc.wins,
                    losses=acc.losses,
                    draws=acc.draws,
                    games_played=acc.games,
                    avg_remaining_pips=acc.avg_remaining_pips,
                    points_scored_avg=acc.avg_points_scored,
                )
            )
        results.leaderboard.sort(key=lambda s: s.elo, reverse=True)

        # Build team leaderboard
        for team_key, tacc in team_stats.items():
            results.team_leaderboard.append(
                TeamStats(
                    team=team_key,
                    wins=tacc.wins,
                    losses=tacc.losses,
                    draws=tacc.draws,
                    games_played=tacc.games,
                )
            )
        results.team_leaderboard.sort(key=lambda t: t.win_rate, reverse=True)

        return results

    def _run_one_game(
        self,
        game_id: int,
        seat_strategies: list[Strategy],
        stats: dict[str, _Accum],
        team_stats: dict[tuple[str, str], _TeamAccum],
    ) -> GameRecord:
        # Team A = seats 0, 2; Team B = seats 1, 3
        team_a = tuple(sorted([seat_strategies[0].name, seat_strategies[2].name]))
        team_b = tuple(sorted([seat_strategies[1].name, seat_strategies[3].name]))

        players = [
            Player(seat=i, strategy=seat_strategies[i])
            for i in range(4)
        ]

        game_rng = random.Random(self._rng.randint(0, 2**63))
        game = Game(players=players, game_mode=GameMode.PAIRS_4P, rng=game_rng)
        result = game.play_round()

        strategy_names = [s.name for s in seat_strategies]

        # Determine winner team
        if not result.winner_seats:
            winner_team = None
        elif 0 in result.winner_seats:
            winner_team = "A"
        else:
            winner_team = "B"

        team_a_pips = result.pip_sums[0] + result.pip_sums[2]
        team_b_pips = result.pip_sums[1] + result.pip_sums[3]

        # Compact move history: [seat, high, low, "l"/"r"] or [seat] for pass
        moves: list[list] = []
        for seat, action in result.move_history:
            if action.tile is None:
                moves.append([seat])
            else:
                end_ch = "l" if action.end == "left" else "r"
                moves.append([seat, action.tile.high, action.tile.low, end_ch])

        # Update Elo: use team_a key vs team_b key
        # We use the individual strategy names for Elo (not teams)
        self._update_elo(seat_strategies, result)

        # Update per-strategy stats
        self._update_stats(result, seat_strategies, stats)

        # Update per-team stats
        self._update_team_stats(team_a, team_b, winner_team, team_stats)

        return GameRecord(
            game_id=game_id,
            strategies=strategy_names,
            team_a=team_a,
            team_b=team_b,
            winner_team=winner_team,
            is_stalemate=result.is_stalemate,
            pip_sums=result.pip_sums,
            team_a_pips=team_a_pips,
            team_b_pips=team_b_pips,
            points=result.points,
            moves=moves,
        )

    def _update_elo(
        self,
        seat_strats: list[Strategy],
        result: 'engine.game.RoundResult',
    ) -> None:
        # In PAIRS_4P, update Elo for each unique strategy on winning vs losing side
        winner_seats = set(result.winner_seats)
        team_a_seats = {0, 2}
        team_b_seats = {1, 3}

        team_a_names = {seat_strats[s].name for s in team_a_seats}
        team_b_names = {seat_strats[s].name for s in team_b_seats}

        # Exclude strategies on both teams (playing against themselves)
        shared = team_a_names & team_b_names
        only_a = list(team_a_names - shared)
        only_b = list(team_b_names - shared)

        # Nothing to update if all strategies are shared
        if not only_a and not only_b:
            return

        if not winner_seats:
            # Draw
            for n1 in only_a:
                for n2 in only_b:
                    self._elo.update(n1, n2, draw=True)
        elif winner_seats & team_a_seats:
            # Team A won
            if only_a and only_b:
                self._elo.update_multiplayer(winners=only_a, losers=only_b)
        else:
            # Team B won
            if only_a and only_b:
                self._elo.update_multiplayer(winners=only_b, losers=only_a)

    def _update_stats(
        self,
        result: 'engine.game.RoundResult',
        seat_strats: list[Strategy],
        stats: dict[str, _Accum],
    ) -> None:
        winner_seats = set(result.winner_seats)
        is_draw = len(winner_seats) == 0

        strat_seats: dict[str, list[int]] = defaultdict(list)
        for seat, strat in enumerate(seat_strats):
            strat_seats[strat.name].append(seat)

        team_a_seats = {0, 2}
        team_b_seats = {1, 3}

        for name, seats in strat_seats.items():
            acc = stats[name]
            acc.games += 1

            seats_set = set(seats)
            on_team_a = bool(seats_set & team_a_seats)
            on_team_b = bool(seats_set & team_b_seats)

            if is_draw or (on_team_a and on_team_b):
                # Strategy on both teams = effective draw
                acc.draws += 1
            elif seats_set & winner_seats:
                acc.wins += 1
            else:
                acc.losses += 1

            total_pips = sum(result.pip_sums[s] for s in seats)
            acc.total_remaining_pips += total_pips

            if result.points > 0 and seats_set & winner_seats and not (on_team_a and on_team_b):
                acc.total_points_scored += result.points

    def _update_team_stats(
        self,
        team_a: tuple[str, str],
        team_b: tuple[str, str],
        winner_team: str | None,
        team_stats: dict[tuple[str, str], _TeamAccum],
    ) -> None:
        # Skip mirror matchups (same composition on both sides)
        if team_a == team_b:
            return

        ta = team_stats[team_a]
        tb = team_stats[team_b]
        ta.games += 1
        tb.games += 1

        if winner_team is None:
            ta.draws += 1
            tb.draws += 1
        elif winner_team == "A":
            ta.wins += 1
            tb.losses += 1
        else:
            tb.wins += 1
            ta.losses += 1


class RoundRobinRunner:
    """Deterministic round-robin: every unique pair of strategies plays as
    Team A vs Team B with homogeneous teams.  Seat assignment is flipped
    halfway through each matchup to control for first-player advantage."""

    def __init__(
        self,
        strategies: list[Strategy],
        games_per_matchup: int = 200,
        rng: random.Random | None = None,
    ) -> None:
        self._strategies = strategies
        self._games_per_matchup = games_per_matchup
        self._rng = rng or random.Random()
        self._elo = EloSystem()
        self._progress_cb: Callable[[int, int], None] | None = None

    def set_progress_callback(self, cb: Callable[[int, int], None]) -> None:
        self._progress_cb = cb

    def run(self) -> SimulationResults:
        results = SimulationResults()
        stats: dict[str, _Accum] = defaultdict(_Accum)
        team_stats: dict[tuple[str, str], _TeamAccum] = defaultdict(_TeamAccum)

        # Build all unique matchups (C(n,2))
        matchups = list(itertools.combinations(self._strategies, 2))
        total_games = len(matchups) * self._games_per_matchup

        elo_snap_interval = max(1, total_games // 500)
        game_id = 0

        for strat_a, strat_b in matchups:
            half = self._games_per_matchup // 2
            for i in range(self._games_per_matchup):
                # First half: strat_a on Team A (seats 0,2), strat_b on Team B (seats 1,3)
                # Second half: swap
                if i < half:
                    seat_strategies = [strat_a, strat_b, strat_a, strat_b]
                else:
                    seat_strategies = [strat_b, strat_a, strat_b, strat_a]

                record = self._run_one_game(game_id, seat_strategies, stats, team_stats)
                results.games.append(record)

                game_num = game_id + 1
                if game_num % elo_snap_interval == 0 or game_num == total_games:
                    for name in self._elo.get_all_ratings():
                        if name not in results.elo_history:
                            results.elo_history[name] = []
                        results.elo_history[name].append(
                            EloSnapshot(game_number=game_num, elo=self._elo.get_rating(name))
                        )

                if self._progress_cb:
                    self._progress_cb(game_num, total_games)

                game_id += 1

        # Build strategy leaderboard
        for strat in self._strategies:
            acc = stats[strat.name]
            results.leaderboard.append(
                StrategyStats(
                    name=strat.name,
                    elo=self._elo.get_rating(strat.name),
                    wins=acc.wins,
                    losses=acc.losses,
                    draws=acc.draws,
                    games_played=acc.games,
                    avg_remaining_pips=acc.avg_remaining_pips,
                    points_scored_avg=acc.avg_points_scored,
                )
            )
        results.leaderboard.sort(key=lambda s: s.elo, reverse=True)

        # Build team leaderboard
        for team_key, tacc in team_stats.items():
            results.team_leaderboard.append(
                TeamStats(
                    team=team_key,
                    wins=tacc.wins,
                    losses=tacc.losses,
                    draws=tacc.draws,
                    games_played=tacc.games,
                )
            )
        results.team_leaderboard.sort(key=lambda t: t.win_rate, reverse=True)

        return results

    # -- internal helpers (reuse same logic as SimulationRunner) --

    def _run_one_game(
        self,
        game_id: int,
        seat_strategies: list[Strategy],
        stats: dict[str, _Accum],
        team_stats: dict[tuple[str, str], _TeamAccum],
    ) -> GameRecord:
        team_a = tuple(sorted([seat_strategies[0].name, seat_strategies[2].name]))
        team_b = tuple(sorted([seat_strategies[1].name, seat_strategies[3].name]))

        players = [
            Player(seat=i, strategy=seat_strategies[i])
            for i in range(4)
        ]

        game_rng = random.Random(self._rng.randint(0, 2**63))
        game = Game(players=players, game_mode=GameMode.PAIRS_4P, rng=game_rng)
        result = game.play_round()

        strategy_names = [s.name for s in seat_strategies]

        if not result.winner_seats:
            winner_team = None
        elif 0 in result.winner_seats:
            winner_team = "A"
        else:
            winner_team = "B"

        team_a_pips = result.pip_sums[0] + result.pip_sums[2]
        team_b_pips = result.pip_sums[1] + result.pip_sums[3]

        moves: list[list] = []
        for seat, action in result.move_history:
            if action.tile is None:
                moves.append([seat])
            else:
                end_ch = "l" if action.end == "left" else "r"
                moves.append([seat, action.tile.high, action.tile.low, end_ch])

        self._update_elo(seat_strategies, result)
        self._update_stats(result, seat_strategies, stats)
        self._update_team_stats(team_a, team_b, winner_team, team_stats)

        return GameRecord(
            game_id=game_id,
            strategies=strategy_names,
            team_a=team_a,
            team_b=team_b,
            winner_team=winner_team,
            is_stalemate=result.is_stalemate,
            pip_sums=result.pip_sums,
            team_a_pips=team_a_pips,
            team_b_pips=team_b_pips,
            points=result.points,
            moves=moves,
        )

    def _update_elo(
        self,
        seat_strats: list[Strategy],
        result: 'engine.game.RoundResult',
    ) -> None:
        winner_seats = set(result.winner_seats)
        team_a_seats = {0, 2}
        team_b_seats = {1, 3}

        team_a_names = {seat_strats[s].name for s in team_a_seats}
        team_b_names = {seat_strats[s].name for s in team_b_seats}

        shared = team_a_names & team_b_names
        only_a = list(team_a_names - shared)
        only_b = list(team_b_names - shared)

        if not only_a and not only_b:
            return

        if not winner_seats:
            for n1 in only_a:
                for n2 in only_b:
                    self._elo.update(n1, n2, draw=True)
        elif winner_seats & team_a_seats:
            if only_a and only_b:
                self._elo.update_multiplayer(winners=only_a, losers=only_b)
        else:
            if only_a and only_b:
                self._elo.update_multiplayer(winners=only_b, losers=only_a)

    def _update_stats(
        self,
        result: 'engine.game.RoundResult',
        seat_strats: list[Strategy],
        stats: dict[str, _Accum],
    ) -> None:
        winner_seats = set(result.winner_seats)
        is_draw = len(winner_seats) == 0

        strat_seats: dict[str, list[int]] = defaultdict(list)
        for seat, strat in enumerate(seat_strats):
            strat_seats[strat.name].append(seat)

        team_a_seats = {0, 2}
        team_b_seats = {1, 3}

        for name, seats in strat_seats.items():
            acc = stats[name]
            acc.games += 1

            seats_set = set(seats)
            on_team_a = bool(seats_set & team_a_seats)
            on_team_b = bool(seats_set & team_b_seats)

            if is_draw or (on_team_a and on_team_b):
                acc.draws += 1
            elif seats_set & winner_seats:
                acc.wins += 1
            else:
                acc.losses += 1

            total_pips = sum(result.pip_sums[s] for s in seats)
            acc.total_remaining_pips += total_pips

            if result.points > 0 and seats_set & winner_seats and not (on_team_a and on_team_b):
                acc.total_points_scored += result.points

    def _update_team_stats(
        self,
        team_a: tuple[str, str],
        team_b: tuple[str, str],
        winner_team: str | None,
        team_stats: dict[tuple[str, str], _TeamAccum],
    ) -> None:
        if team_a == team_b:
            return

        ta = team_stats[team_a]
        tb = team_stats[team_b]
        ta.games += 1
        tb.games += 1

        if winner_team is None:
            ta.draws += 1
            tb.draws += 1
        elif winner_team == "A":
            ta.wins += 1
            tb.losses += 1
        else:
            tb.wins += 1
            ta.losses += 1


class _Accum:
    __slots__ = ("wins", "losses", "draws", "games", "total_remaining_pips", "total_points_scored")

    def __init__(self) -> None:
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.games = 0
        self.total_remaining_pips = 0
        self.total_points_scored = 0

    @property
    def avg_remaining_pips(self) -> float:
        return self.total_remaining_pips / self.games if self.games else 0.0

    @property
    def avg_points_scored(self) -> float:
        return self.total_points_scored / self.games if self.games else 0.0


class _TeamAccum:
    __slots__ = ("wins", "losses", "draws", "games")

    def __init__(self) -> None:
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.games = 0
