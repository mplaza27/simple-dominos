from __future__ import annotations

import random

from engine.game import Action, GameState
from strategies.base import Strategy


class GreedyDoublesStrategy(Strategy):
    """Like Greedy, but prioritizes playing doubles first. Among doubles
    (or among non-doubles if no doubles are playable), picks the highest pip sum."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def choose_action(self, state: GameState) -> Action:
        actions = state.valid_actions
        if len(actions) == 1:
            return actions[0]

        play_actions = [a for a in actions if a.tile is not None]
        if not play_actions:
            return actions[0]

        # Prefer doubles
        doubles = [a for a in play_actions if a.tile is not None and a.tile.is_double]
        pool = doubles if doubles else play_actions

        max_pips = max(a.tile.pip_sum for a in pool)  # type: ignore[union-attr]
        best = [a for a in pool if a.tile is not None and a.tile.pip_sum == max_pips]
        return self._rng.choice(best)
