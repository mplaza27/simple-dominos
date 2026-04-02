from __future__ import annotations

import random

from engine.game import Action, GameState
from strategies.base import Strategy


class NeverPassedStrategy(Strategy):
    """Optimizes for never having to pass. After playing a tile, scores the
    remaining hand by how many distinct pip values (0-9) it still covers.
    Picks the action that maximizes that coverage, so future board ends are
    more likely to match something in hand. Tiebreaks by highest pip sum."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def _remaining_coverage(self, state: GameState, action: Action) -> int:
        """Count distinct pip values in hand after playing the given tile."""
        assert action.tile is not None
        values: set[int] = set()
        for tile in state.hand:
            if tile == action.tile:
                continue
            values.add(tile.high)
            values.add(tile.low)
        return len(values)

    def choose_action(self, state: GameState) -> Action:
        actions = state.valid_actions
        if len(actions) == 1:
            return actions[0]

        play_actions = [a for a in actions if a.tile is not None]
        if not play_actions:
            return actions[0]

        max_coverage = max(self._remaining_coverage(state, a) for a in play_actions)
        best = [a for a in play_actions if self._remaining_coverage(state, a) == max_coverage]

        # Tiebreak: highest pip sum (shed points when coverage is equal)
        max_pips = max(a.tile.pip_sum for a in best)  # type: ignore[union-attr]
        final = [a for a in best if a.tile is not None and a.tile.pip_sum == max_pips]
        return self._rng.choice(final)
