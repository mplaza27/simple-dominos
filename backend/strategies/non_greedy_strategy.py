from __future__ import annotations

import random

from engine.game import Action, GameState
from strategies.base import Strategy


class NonGreedyStrategy(Strategy):
    """Plays the lowest pip-sum tile first. Sheds cheap tiles early."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def choose_action(self, state: GameState) -> Action:
        actions = state.valid_actions
        if len(actions) == 1:
            return actions[0]

        play_actions = [a for a in actions if a.tile is not None]
        if not play_actions:
            return actions[0]

        min_pips = min(a.tile.pip_sum for a in play_actions)  # type: ignore[union-attr]
        best = [a for a in play_actions if a.tile is not None and a.tile.pip_sum == min_pips]
        return self._rng.choice(best)
