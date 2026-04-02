from __future__ import annotations

import random

from engine.game import Action, GameState
from engine.tile import Board, Tile
from strategies.base import Strategy


class PassTrackerStrategy(Strategy):
    """Tracks what board ends were showing when opponents passed, inferring
    values they lack. Prefers playing values opponents can't match.
    Falls back to highest pip sum when no exploitable info exists."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def _infer_missing_values(self, state: GameState) -> dict[int, set[int]]:
        """For each opponent seat, return the set of pip values they likely lack,
        based on board ends at the time they passed."""
        missing: dict[int, set[int]] = {}

        # Replay the board from move history to know ends at each point
        board = Board()
        for seat, action in state.move_history:
            if action.is_pass:
                ends = board.ends
                if ends is not None and seat != state.current_seat:
                    if seat not in missing:
                        missing[seat] = set()
                    missing[seat].add(ends[0])
                    missing[seat].add(ends[1])
            elif action.tile is not None and action.end is not None:
                board.play(action.tile, action.end)

        return missing

    def choose_action(self, state: GameState) -> Action:
        actions = state.valid_actions
        if len(actions) == 1:
            return actions[0]

        play_actions = [a for a in actions if a.tile is not None]
        if not play_actions:
            return actions[0]

        missing = self._infer_missing_values(state)
        if not missing:
            # No pass info yet — fall back to greedy
            return self._pick_highest_pip(play_actions)

        # All values opponents are known to lack
        all_missing = set()
        for vals in missing.values():
            all_missing.update(vals)

        # Score each action: prefer plays that leave a board end the opponent can't match.
        # i.e., after playing tile on end, the new exposed value is one opponents lack.
        def exploit_score(action: Action) -> int:
            assert action.tile is not None and action.end is not None
            tile = action.tile
            # What value will be the new exposed end after this play?
            if state.board_ends is None:
                # First move — both ends exposed
                new_vals = {tile.high, tile.low}
            elif action.end == "left":
                new_vals = {tile.other_value(state.board_ends[0])}
            else:
                new_vals = {tile.other_value(state.board_ends[1])}

            return len(new_vals & all_missing)

        max_score = max(exploit_score(a) for a in play_actions)

        if max_score > 0:
            best = [a for a in play_actions if exploit_score(a) == max_score]
            return self._pick_highest_pip(best)

        # No exploitable plays — fall back to greedy
        return self._pick_highest_pip(play_actions)

    def _pick_highest_pip(self, actions: list[Action]) -> Action:
        max_pips = max(a.tile.pip_sum for a in actions)  # type: ignore[union-attr]
        best = [a for a in actions if a.tile is not None and a.tile.pip_sum == max_pips]
        return self._rng.choice(best)
