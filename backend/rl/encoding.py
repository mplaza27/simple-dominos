from __future__ import annotations

import torch

from engine.game import Action, GameState
from engine.tile import Board, Tile

# ── Constants ─────────────────────────────────────────────────────────────────
# State vector layout (total 511 dims):
#   [0:55]     hand tiles — binary indicator for each of 55 tiles
#   [55:110]   played tiles on board — binary indicator
#   [110:120]  board left end — one-hot over values 0-9
#   [120:130]  board right end — one-hot over values 0-9
#   [130]      board empty flag
#   [131:134]  opponent tile counts (3 opponents in 4P), normalized by 10
#   [134:138]  last-action pass flag per seat (most recent move only)
#   [138]      is_first_move
#   [139:143]  current seat — one-hot
#   [143:147]  teammate seat — one-hot (zeros if no teammate)
#   --- history-derived features (require move_history) ---
#   [147:202]  tiles played by seat 0 — binary (55-dim)
#   [202:257]  tiles played by seat 1 — binary (55-dim)
#   [257:312]  tiles played by seat 2 — binary (55-dim)
#   [312:367]  tiles played by seat 3 — binary (55-dim)
#   [367:371]  cumulative pass count per seat, normalized by 10
#   [371:426]  tile play order — for each of 55 tiles: (turn_index+1)/50 when played, 0 if unplayed
#   --- opponent inference features ---
#   [426:456]  inferred missing pip values per other seat (3 seats × 10 pips, binary)
#              seats in relative order: (current+1)%4, (current+2)%4, (current+3)%4
#   [456:459]  opponent threat estimate per other seat (3 dims): 1 - missing_count/10
#   --- v2 features ---
#   [459:469]  pip value counts — fraction of tiles containing pip V that are played (10 dims)
#   [469:471]  end scarcity — fraction of tiles matching each board end that are played (2 dims)
#   [471:511]  per-seat pip value play counts (4 seats x 10 pips = 40 dims), normalized by 10

STATE_DIM = 511

# Normalization constant for turn index; covers a full game including passes
_MAX_TURNS = 50

# Action vector layout (total 111 dims):
#   tile_idx * 2 + 0  → play tile on left end   (indices 0..109, even)
#   tile_idx * 2 + 1  → play tile on right end  (indices 1..109, odd)
#   110               → pass
ACTION_DIM = 111
PASS_IDX = 110

_TILES_PER_PLAYER = 10


def tile_to_idx(tile: Tile) -> int:
    """Canonical index 0-54 for a Tile.  tile.high >= tile.low always."""
    return tile.high * (tile.high + 1) // 2 + tile.low


def action_to_idx(action: Action) -> int:
    if action.is_pass:
        return PASS_IDX
    assert action.tile is not None and action.end is not None
    return tile_to_idx(action.tile) * 2 + (0 if action.end == "left" else 1)


def encode_state(state: GameState) -> torch.Tensor:
    """Encode a GameState into a float32 tensor of shape (STATE_DIM,)."""
    parts: list[torch.Tensor] = []

    # 1. Hand tiles (55-dim binary)
    hand_vec = torch.zeros(55)
    for tile in state.hand:
        hand_vec[tile_to_idx(tile)] = 1.0
    parts.append(hand_vec)

    # 2. Played tiles (55-dim binary)
    played_vec = torch.zeros(55)
    for tile in state.played_tiles:
        played_vec[tile_to_idx(tile)] = 1.0
    parts.append(played_vec)

    # 3-4. Board ends one-hot (10-dim each), 5. Board empty flag
    left_vec = torch.zeros(10)
    right_vec = torch.zeros(10)
    empty_vec = torch.zeros(1)
    if state.board_ends is None:
        empty_vec[0] = 1.0
    else:
        left_vec[state.board_ends[0]] = 1.0
        right_vec[state.board_ends[1]] = 1.0
    parts.extend([left_vec, right_vec, empty_vec])

    # 6. Opponent tile counts normalized (3-dim; 4P has 3 opponents)
    opp_vec = torch.zeros(3)
    for i, count in enumerate(state.opponent_tile_counts[:3]):
        opp_vec[i] = count / _TILES_PER_PLAYER
    parts.append(opp_vec)

    # 7. Pass history per seat (4-dim)
    pass_vec = torch.zeros(4)
    for i, passed in enumerate(state.pass_history[:4]):
        pass_vec[i] = 1.0 if passed else 0.0
    parts.append(pass_vec)

    # 8. Is first move (1-dim)
    parts.append(torch.tensor([1.0 if state.is_first_move else 0.0]))

    # 9. Current seat one-hot (4-dim)
    seat_vec = torch.zeros(4)
    if 0 <= state.current_seat < 4:
        seat_vec[state.current_seat] = 1.0
    parts.append(seat_vec)

    # 10. Teammate seat one-hot (4-dim; zeros if no teammate)
    teammate_vec = torch.zeros(4)
    if state.teammate_seat is not None and 0 <= state.teammate_seat < 4:
        teammate_vec[state.teammate_seat] = 1.0
    parts.append(teammate_vec)

    # 11-13. Single pass through move_history to derive three features:
    #   - Per-seat played tiles (4 × 55-dim binary)
    #   - Cumulative pass count per seat, normalized by 10 (4-dim)
    #   - Tile play order: for each tile, normalized turn index when played (0 if unplayed)
    #   Also replay board state for opponent pass inference (feature 14-15).
    seat_played: list[torch.Tensor] = [torch.zeros(55) for _ in range(4)]
    pass_counts = torch.zeros(4)
    play_order = torch.zeros(55)
    # Track inferred missing pip values per seat (for opponent inference)
    missing_values: dict[int, set[int]] = {}
    board = Board()
    for turn_idx, (seat, action) in enumerate(state.move_history):
        if action.is_pass:
            if 0 <= seat < 4:
                pass_counts[seat] += 1.0
            # Infer missing values: when a seat passes, they lack the board ends
            ends = board.ends
            if ends is not None and seat != state.current_seat:
                if seat not in missing_values:
                    missing_values[seat] = set()
                missing_values[seat].add(ends[0])
                missing_values[seat].add(ends[1])
        elif action.tile is not None:
            idx = tile_to_idx(action.tile)
            if 0 <= seat < 4:
                seat_played[seat][idx] = 1.0
            play_order[idx] = (turn_idx + 1) / _MAX_TURNS
            if action.end is not None:
                board.play(action.tile, action.end)
    pass_counts /= 10.0
    parts.extend(seat_played)
    parts.append(pass_counts)
    parts.append(play_order)

    # 14. Inferred missing pip values per other seat (3 × 10 = 30 dims, binary)
    # Relative seat order: (current+1)%4, (current+2)%4, (current+3)%4
    cur = state.current_seat
    for offset in (1, 2, 3):
        other_seat = (cur + offset) % 4
        missing_vec = torch.zeros(10)
        for pip_val in missing_values.get(other_seat, ()):
            if 0 <= pip_val <= 9:
                missing_vec[pip_val] = 1.0
        parts.append(missing_vec)

    # 15. Opponent threat estimate (3 dims): 1 - missing_count / 10
    threat_vec = torch.zeros(3)
    for i, offset in enumerate((1, 2, 3)):
        other_seat = (cur + offset) % 4
        n_missing = len(missing_values.get(other_seat, ()))
        threat_vec[i] = 1.0 - min(n_missing, 10) / 10.0
    parts.append(threat_vec)

    # 16. Pip value counts (10 dims): fraction of played tiles containing each pip 0-9
    # Every pip value appears on exactly 10 tiles in a double-9 set.
    pip_counts = torch.zeros(10)
    for tile in state.played_tiles:
        pip_counts[tile.high] += 1.0
        pip_counts[tile.low] += 1.0
        if tile.high == tile.low:
            # Doubles were counted twice above; correct to count once
            pip_counts[tile.high] -= 1.0
    pip_counts /= 10.0
    parts.append(pip_counts)

    # 17. End scarcity (2 dims): fraction of tiles matching each board end that are played
    end_scarcity = torch.zeros(2)
    if state.board_ends is not None:
        for side_idx, end_val in enumerate(state.board_ends):
            count = 0
            for tile in state.played_tiles:
                if tile.high == end_val or tile.low == end_val:
                    count += 1
            end_scarcity[side_idx] = count / 10.0
    parts.append(end_scarcity)

    # 18. Per-seat pip value play counts (4 seats x 10 pips = 40 dims)
    seat_pip_counts = torch.zeros(4, 10)
    for seat, action in state.move_history:
        if not action.is_pass and action.tile is not None and 0 <= seat < 4:
            seat_pip_counts[seat, action.tile.high] += 1.0
            seat_pip_counts[seat, action.tile.low] += 1.0
            if action.tile.high == action.tile.low:
                seat_pip_counts[seat, action.tile.high] -= 1.0
    seat_pip_counts /= 10.0
    parts.append(seat_pip_counts.flatten())

    result = torch.cat(parts)
    assert result.shape == (STATE_DIM,), f"Expected {STATE_DIM}, got {result.shape}"
    return result


def encode_action_mask(valid_actions: list[Action]) -> torch.Tensor:
    """Bool tensor of shape (ACTION_DIM,) — True where action is valid."""
    mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
    for action in valid_actions:
        mask[action_to_idx(action)] = True
    return mask


def decode_action(action_idx: int, valid_actions: list[Action]) -> Action:
    """Map an action index back to the matching Action from valid_actions."""
    for action in valid_actions:
        if action_to_idx(action) == action_idx:
            return action
    raise ValueError(
        f"Action index {action_idx} not found in valid_actions "
        f"{[action_to_idx(a) for a in valid_actions]}"
    )
