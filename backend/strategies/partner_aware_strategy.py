from __future__ import annotations

import random

from engine.game import Action, GameState
from engine.tile import Board, Tile
from engine.types import GameMode
from strategies.base import Strategy


class PartnerAwareStrategy(Strategy):
    """4P Pairs strategy that reads partner signals:
    - If partner opened, favor playing values from their opening tile.
    - If partner passed, try to play (remove) the values they lack.
    - If partner played on an end, play on the opposite end to preserve
      their options.
    Falls back to greedy in non-pairs modes."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def _analyze_partner(self, state: GameState) -> tuple[set[int], set[int], str | None]:
        """Returns (partner_values, partner_missing, partner_last_end).
        - partner_values: pip values the partner has played (they like these).
        - partner_missing: pip values the partner passed on (they lack these).
        - partner_last_end: 'left'/'right'/None — which end partner last played on."""
        partner_values: set[int] = set()
        partner_missing: set[int] = set()
        partner_last_end: str | None = None
        teammate = state.teammate_seat

        if teammate is None:
            return partner_values, partner_missing, partner_last_end

        board = Board()
        for seat, action in state.move_history:
            if seat == teammate:
                if action.is_pass:
                    ends = board.ends
                    if ends is not None:
                        partner_missing.add(ends[0])
                        partner_missing.add(ends[1])
                elif action.tile is not None and action.end is not None:
                    partner_values.add(action.tile.high)
                    partner_values.add(action.tile.low)
                    partner_last_end = action.end

            if not action.is_pass and action.tile is not None and action.end is not None:
                board.play(action.tile, action.end)

        return partner_values, partner_missing, partner_last_end

    def choose_action(self, state: GameState) -> Action:
        actions = state.valid_actions
        if len(actions) == 1:
            return actions[0]

        play_actions = [a for a in actions if a.tile is not None]
        if not play_actions:
            return actions[0]

        # Non-pairs mode: fall back to greedy
        if state.game_mode != GameMode.PAIRS_4P or state.teammate_seat is None:
            return self._pick_highest_pip(play_actions)

        partner_values, partner_missing, partner_last_end = self._analyze_partner(state)

        def score(action: Action) -> float:
            assert action.tile is not None and action.end is not None
            tile = action.tile
            s = 0.0

            # Reward playing tiles with values partner has shown (they can follow up)
            for v in partner_values:
                if tile.has_value(v):
                    s += 3.0
                    break

            # Reward playing values partner is missing (clear the board of those)
            if state.board_ends is not None:
                if action.end == "left":
                    new_end = tile.other_value(state.board_ends[0])
                else:
                    new_end = tile.other_value(state.board_ends[1])

                # If the new end is a value partner lacks, that's bad for partner
                if new_end in partner_missing:
                    s -= 2.0
                # If the new end is a value partner has played, good for partner
                if new_end in partner_values:
                    s += 2.0

            # Play opposite end from partner to preserve their side
            if partner_last_end is not None:
                opposite = "right" if partner_last_end == "left" else "left"
                if action.end == opposite:
                    s += 1.0

            # Small tiebreaker: higher pip sum
            s += tile.pip_sum / 100.0

            return s

        max_score = max(score(a) for a in play_actions)
        best = [a for a in play_actions if score(a) == max_score]
        return self._rng.choice(best)

    def _pick_highest_pip(self, actions: list[Action]) -> Action:
        max_pips = max(a.tile.pip_sum for a in actions)  # type: ignore[union-attr]
        best = [a for a in actions if a.tile is not None and a.tile.pip_sum == max_pips]
        return self._rng.choice(best)
