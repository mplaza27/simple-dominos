from __future__ import annotations

import random

from engine.game import Action, GameState
from strategies.base import Strategy


class GreedyStrategy(Strategy):
    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def choose_action(self, state: GameState) -> Action:
        actions = state.valid_actions
        # If only option is pass, take it
        if len(actions) == 1:
            return actions[0]
        # Filter out pass actions, pick highest pip sum
        play_actions = [a for a in actions if a.tile is not None]
        if not play_actions:
            return actions[0]
        max_pips = max(a.tile.pip_sum for a in play_actions)  # type: ignore[union-attr]
        best = [a for a in play_actions if a.tile is not None and a.tile.pip_sum == max_pips]
        return self._rng.choice(best)
