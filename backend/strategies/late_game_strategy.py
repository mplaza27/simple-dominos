from __future__ import annotations

import random

from engine.game import Action, GameState
from strategies.base import Strategy


class LateGameStrategy(Strategy):
    """Keeps well-connected tiles (especially doubles) for later by playing
    the least-connected tile first. A tile's connectivity is how many other
    tiles in hand share a pip value with it. Doubles count double since both
    sides connect to the same value."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def _connectivity(self, state: GameState, action: Action) -> int:
        """Count how many *other* tiles in hand share a pip value with this tile."""
        assert action.tile is not None
        tile = action.tile
        count = 0
        for other in state.hand:
            if other == tile:
                continue
            if other.has_value(tile.high) or other.has_value(tile.low):
                count += 1
        # Doubles are extra valuable connectors — boost their score
        if tile.is_double:
            count += 2
        return count

    def choose_action(self, state: GameState) -> Action:
        actions = state.valid_actions
        if len(actions) == 1:
            return actions[0]

        play_actions = [a for a in actions if a.tile is not None]
        if not play_actions:
            return actions[0]

        # Play the tile with the LOWEST connectivity (save connectors for later)
        min_conn = min(self._connectivity(state, a) for a in play_actions)
        least_connected = [a for a in play_actions if self._connectivity(state, a) == min_conn]

        # Among equally low connectivity, prefer higher pip sum to shed points
        max_pips = max(a.tile.pip_sum for a in least_connected)  # type: ignore[union-attr]
        best = [a for a in least_connected if a.tile is not None and a.tile.pip_sum == max_pips]
        return self._rng.choice(best)
