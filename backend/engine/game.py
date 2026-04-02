from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from engine.player import Player
from engine.tile import Board, Tile, create_full_set
from engine.types import End, GameMode, InvalidMoveError


@dataclass(frozen=True, slots=True)
class Action:
    tile: Tile | None
    end: End | None

    @classmethod
    def pass_action(cls) -> Action:
        return cls(tile=None, end=None)

    @classmethod
    def play(cls, tile: Tile, end: End) -> Action:
        return cls(tile=tile, end=end)

    @property
    def is_pass(self) -> bool:
        return self.tile is None

    def __repr__(self) -> str:
        if self.is_pass:
            return "Action(PASS)"
        return f"Action({self.tile} -> {self.end})"


@dataclass(frozen=True, slots=True)
class GameState:
    hand: tuple[Tile, ...]
    board_ends: tuple[int, int] | None
    played_tiles: frozenset[Tile]
    opponent_tile_counts: tuple[int, ...]
    pass_history: tuple[bool, ...]
    valid_actions: list[Action]
    current_seat: int
    teammate_seat: int | None
    game_mode: GameMode
    is_first_move: bool
    move_history: tuple[tuple[int, Action], ...]
    num_players: int


@dataclass(frozen=True, slots=True)
class RoundResult:
    winner_seats: list[int]
    is_stalemate: bool
    final_hands: dict[int, list[Tile]]
    pip_sums: dict[int, int]
    points: int
    move_history: list[tuple[int, Action]]


_PLAYER_COUNTS: dict[GameMode, int] = {
    GameMode.FFA_2P: 2,
    GameMode.FFA_3P: 3,
    GameMode.PAIRS_4P: 4,
}

TILES_PER_PLAYER = 10


class Game:
    def __init__(
        self,
        players: list[Player],
        game_mode: GameMode,
        rng: random.Random | None = None,
    ) -> None:
        expected = _PLAYER_COUNTS[game_mode]
        if len(players) != expected:
            raise ValueError(
                f"{game_mode.value} requires {expected} players, got {len(players)}"
            )

        self._players = players
        self._game_mode = game_mode
        self._rng = rng or random.Random()
        self._board = Board()
        self._move_history: list[tuple[int, Action]] = []
        self._consecutive_passes = 0
        self._game_over = False

        self._deal()
        self._current_index = self._determine_first_player()
        self._is_first_move = True

    def _deal(self) -> None:
        tiles = create_full_set()
        self._rng.shuffle(tiles)
        for player in self._players:
            player.hand = tiles[:TILES_PER_PLAYER]
            tiles = tiles[TILES_PER_PLAYER:]
        # Remaining tiles are out of play — intentionally discarded

    def _determine_first_player(self) -> int:
        # Find highest double across all hands
        best_double: Tile | None = None
        best_seat = 0
        for player in self._players:
            for tile in player.hand:
                if tile.is_double:
                    if best_double is None or tile.pip_sum > best_double.pip_sum:
                        best_double = tile
                        best_seat = player.seat

        if best_double is not None:
            return best_seat

        # No doubles: highest pip-sum tile
        best_tile: Tile | None = None
        best_seat = 0
        for player in self._players:
            for tile in player.hand:
                if best_tile is None or tile.pip_sum > best_tile.pip_sum:
                    best_tile = tile
                    best_seat = player.seat
                elif tile.pip_sum == best_tile.pip_sum and tile.high > best_tile.high:
                    best_tile = tile
                    best_seat = player.seat
        return best_seat

    @property
    def current_player(self) -> Player:
        return self._players[self._current_index]

    @property
    def game_over(self) -> bool:
        return self._game_over

    def _compute_valid_actions(self, player: Player) -> list[Action]:
        if self._board.is_empty:
            # First move: any tile, played to "left" (only one end)
            actions: list[Action] = []
            for tile in player.hand:
                actions.append(Action.play(tile, "left"))
            return actions

        ends = self._board.ends
        assert ends is not None
        left_end, right_end = ends

        actions = []
        for tile in player.hand:
            placements = self._board.valid_placements(tile)
            for end in placements:
                actions.append(Action.play(tile, end))

        if not actions:
            actions.append(Action.pass_action())

        return actions

    def _build_game_state(self, player: Player) -> GameState:
        valid_actions = self._compute_valid_actions(player)

        opponent_counts = tuple(
            p.tile_count for p in self._players if p.seat != player.seat
        )

        # Pass history: one bool per player for their most recent action
        num_players = len(self._players)
        recent_passes: list[bool] = [False] * num_players
        for seat, action in reversed(self._move_history):
            if recent_passes[seat] is False and action.is_pass:
                recent_passes[seat] = True
            # Only look at each player's most recent move
            seen = set()
            all_seen = True
            for s, _ in reversed(self._move_history):
                seen.add(s)
                if len(seen) == num_players:
                    break
            else:
                all_seen = False

        # Simpler: track last action per player
        last_action: dict[int, Action] = {}
        for seat, action in reversed(self._move_history):
            if seat not in last_action:
                last_action[seat] = action
            if len(last_action) == num_players:
                break
        pass_hist = tuple(
            last_action.get(i, Action.play(Tile(0, 0), "left")).is_pass
            for i in range(num_players)
        )

        teammate: int | None = None
        if self._game_mode == GameMode.PAIRS_4P:
            teammate = (player.seat + 2) % 4

        return GameState(
            hand=tuple(player.hand),
            board_ends=self._board.ends,
            played_tiles=self._board.played_tiles,
            opponent_tile_counts=opponent_counts,
            pass_history=pass_hist,
            valid_actions=valid_actions,
            current_seat=player.seat,
            teammate_seat=teammate,
            game_mode=self._game_mode,
            is_first_move=self._is_first_move,
            move_history=tuple(self._move_history),
            num_players=len(self._players),
        )

    def _validate_and_apply(self, player: Player, action: Action) -> None:
        valid_actions = self._compute_valid_actions(player)

        if action not in valid_actions:
            raise InvalidMoveError(
                f"Action {action} is not valid. Valid actions: {valid_actions}"
            )

        if action.is_pass:
            self._consecutive_passes += 1
            self._move_history.append((player.seat, action))
            return

        assert action.tile is not None and action.end is not None
        self._board.play(action.tile, action.end)
        player.remove_tile(action.tile)
        self._consecutive_passes = 0
        self._move_history.append((player.seat, action))
        self._is_first_move = False

    def _advance_turn(self) -> None:
        self._current_index = (self._current_index + 1) % len(self._players)

    def _check_game_over(self) -> bool:
        # Win: current player emptied hand
        if not self.current_player.has_tiles:
            return True
        # Stalemate: all players passed consecutively
        if self._consecutive_passes >= len(self._players):
            return True
        return False

    def step(self) -> tuple[int, Action, bool]:
        if self._game_over:
            raise RuntimeError("Game is already over")

        player = self.current_player
        state = self._build_game_state(player)
        action = player.strategy.choose_action(state)
        self._validate_and_apply(player, action)

        seat = player.seat

        if self._check_game_over():
            self._game_over = True
            return seat, action, True

        self._advance_turn()
        return seat, action, False

    def play_round(
        self, max_moves: int = 0, max_seconds: float = 0,
    ) -> RoundResult:
        for p in self._players:
            p.strategy.on_game_start()

        move_count = 0
        start_time = time.monotonic() if max_seconds else 0.0
        while not self._game_over:
            self.step()
            move_count += 1
            if max_moves and move_count >= max_moves:
                self._game_over = True
                self._consecutive_passes = len(self._players)  # force stalemate
            if max_seconds and time.monotonic() - start_time >= max_seconds:
                self._game_over = True
                self._consecutive_passes = len(self._players)  # force stalemate

        for p in self._players:
            p.strategy.on_game_end()

        return self._build_result()

    def _build_result(self) -> RoundResult:
        final_hands = {p.seat: list(p.hand) for p in self._players}
        pip_sums = {p.seat: p.pip_sum for p in self._players}

        # Check for empty-hand winner
        empty_hand_winners = [p.seat for p in self._players if not p.has_tiles]

        if empty_hand_winners:
            is_stalemate = False
            if self._game_mode == GameMode.PAIRS_4P:
                winner_seat = empty_hand_winners[0]
                winner_team = {winner_seat, (winner_seat + 2) % 4}
                loser_team = {s for s in range(4)} - winner_team
                points = sum(pip_sums[s] for s in loser_team)
                return RoundResult(
                    winner_seats=sorted(winner_team),
                    is_stalemate=False,
                    final_hands=final_hands,
                    pip_sums=pip_sums,
                    points=points,
                    move_history=list(self._move_history),
                )
            else:
                return RoundResult(
                    winner_seats=empty_hand_winners,
                    is_stalemate=False,
                    final_hands=final_hands,
                    pip_sums=pip_sums,
                    points=0,
                    move_history=list(self._move_history),
                )

        # Stalemate
        is_stalemate = True

        if self._game_mode == GameMode.PAIRS_4P:
            team_a = {0, 2}
            team_b = {1, 3}

            # Individual player with lowest pip sum wins for their team
            min_pips = min(pip_sums.values())
            lowest_seats = [s for s, p in pip_sums.items() if p == min_pips]

            # Check which teams have the lowest individual
            a_has_lowest = any(s in team_a for s in lowest_seats)
            b_has_lowest = any(s in team_b for s in lowest_seats)

            if a_has_lowest and not b_has_lowest:
                winner_seats = sorted(team_a)
                loser_pips = sum(pip_sums[s] for s in team_b)
                points = loser_pips
            elif b_has_lowest and not a_has_lowest:
                winner_seats = sorted(team_b)
                loser_pips = sum(pip_sums[s] for s in team_a)
                points = loser_pips
            else:
                winner_seats = []  # Tie — both teams have a player at min
                points = 0

            return RoundResult(
                winner_seats=winner_seats,
                is_stalemate=True,
                final_hands=final_hands,
                pip_sums=pip_sums,
                points=points,
                move_history=list(self._move_history),
            )
        else:
            # FFA: lowest individual pip sum wins
            min_pips = min(pip_sums.values())
            winner_seats = [s for s, p in pip_sums.items() if p == min_pips]

            return RoundResult(
                winner_seats=sorted(winner_seats),
                is_stalemate=True,
                final_hands=final_hands,
                pip_sums=pip_sums,
                points=0,
                move_history=list(self._move_history),
            )
