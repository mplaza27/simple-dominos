from __future__ import annotations

import random

from engine.game import Action, GameState
from strategies.base import Strategy


class RandomStrategy(Strategy):
    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def choose_action(self, state: GameState) -> Action:
        return self._rng.choice(state.valid_actions)
