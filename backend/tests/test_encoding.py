"""Tests for rl/encoding.py."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from engine.game import Action, GameState
from engine.tile import Tile
from engine.types import GameMode
from rl.encoding import STATE_DIM, _MAX_TURNS, encode_state


def _make_state(move_history: tuple[tuple[int, Action], ...]) -> GameState:
    """Minimal GameState with the given move_history."""
    return GameState(
        hand=(Tile(0, 0),),
        board_ends=None,
        played_tiles=frozenset(),
        opponent_tile_counts=(10, 10, 10),
        pass_history=(False, False, False, False),
        valid_actions=[],
        current_seat=0,
        teammate_seat=2,
        game_mode=GameMode.PAIRS_4P,
        is_first_move=False,
        move_history=move_history,
        num_players=4,
    )


def test_output_shape_no_history() -> None:
    state = _make_state(())
    result = encode_state(state)
    assert result.shape == (STATE_DIM,)
    assert STATE_DIM == 511


def test_output_shape_with_history() -> None:
    t = Tile(3, 5)
    history: tuple[tuple[int, Action], ...] = (
        (0, Action.play(t, "left")),
        (1, Action.pass_action()),
    )
    state = _make_state(history)
    result = encode_state(state)
    assert result.shape == (511,)


def test_play_order_unplayed_is_zero() -> None:
    state = _make_state(())
    result = encode_state(state)
    play_order = result[371:426]
    assert play_order.sum().item() == 0.0


def test_play_order_records_turn_index() -> None:
    """Tile played at turn 0 gets value 1/50; tile at turn 2 gets 3/50."""
    t0 = Tile(0, 1)  # turn 0: board ends become left=0, right=1
    t2 = Tile(1, 2)  # turn 2: plays on right (matches 1), board right=2

    history: tuple[tuple[int, Action], ...] = (
        (0, Action.play(t0, "left")),
        (1, Action.pass_action()),
        (2, Action.play(t2, "right")),
    )
    state = _make_state(history)
    result = encode_state(state)

    from rl.encoding import tile_to_idx
    idx0 = tile_to_idx(t0)
    idx2 = tile_to_idx(t2)

    assert abs(result[371 + idx0].item() - 1 / _MAX_TURNS) < 1e-6
    assert abs(result[371 + idx2].item() - 3 / _MAX_TURNS) < 1e-6
