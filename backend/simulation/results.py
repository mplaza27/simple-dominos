from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EloSnapshot:
    game_number: int
    elo: float


@dataclass(frozen=True, slots=True)
class StrategyStats:
    name: str
    elo: float
    wins: int
    losses: int
    draws: int
    games_played: int
    avg_remaining_pips: float
    points_scored_avg: float

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played


@dataclass(frozen=True, slots=True)
class TeamStats:
    team: tuple[str, str]  # sorted strategy names
    wins: int
    losses: int
    draws: int
    games_played: int

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played


@dataclass(frozen=True, slots=True)
class GameRecord:
    game_id: int
    strategies: list[str]  # strategy name per seat (4 seats)
    team_a: tuple[str, str]  # (seat0_strat, seat2_strat) sorted
    team_b: tuple[str, str]  # (seat1_strat, seat3_strat) sorted
    winner_team: str | None  # "A", "B", or None (draw)
    is_stalemate: bool
    pip_sums: dict[int, int]
    team_a_pips: int
    team_b_pips: int
    points: int
    moves: list[list]  # compact: [seat, high, low, "l"/"r"] or [seat] for pass


@dataclass
class SimulationResults:
    leaderboard: list[StrategyStats] = field(default_factory=list)
    team_leaderboard: list[TeamStats] = field(default_factory=list)
    elo_history: dict[str, list[EloSnapshot]] = field(default_factory=dict)
    games: list[GameRecord] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "leaderboard": [
                {
                    "name": s.name,
                    "elo": round(s.elo, 1),
                    "win_rate": round(s.win_rate, 4),
                    "wins": s.wins,
                    "losses": s.losses,
                    "draws": s.draws,
                    "games_played": s.games_played,
                    "avg_remaining_pips": round(s.avg_remaining_pips, 2),
                    "points_scored_avg": round(s.points_scored_avg, 2),
                }
                for s in self.leaderboard
            ],
            "team_leaderboard": [
                {
                    "team": list(t.team),
                    "wins": t.wins,
                    "losses": t.losses,
                    "draws": t.draws,
                    "games_played": t.games_played,
                    "win_rate": round(t.win_rate, 4),
                }
                for t in self.team_leaderboard
            ],
            "elo_history": {
                name: [{"game_number": s.game_number, "elo": round(s.elo, 1)} for s in snaps]
                for name, snaps in self.elo_history.items()
            },
            "games": [
                {
                    "game_id": g.game_id,
                    "strategies": g.strategies,
                    "team_a": list(g.team_a),
                    "team_b": list(g.team_b),
                    "winner_team": g.winner_team,
                    "is_stalemate": g.is_stalemate,
                    "pip_sums": {str(k): v for k, v in g.pip_sums.items()},
                    "team_a_pips": g.team_a_pips,
                    "team_b_pips": g.team_b_pips,
                    "points": g.points,
                    "moves": g.moves,
                }
                for g in self.games
            ],
            "total_games": len(self.games),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json()))
