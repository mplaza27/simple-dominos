from __future__ import annotations

import random

import pytest

from engine.game import Action, Game, GameState, RoundResult
from engine.player import Player
from engine.tile import Board, Tile, create_full_set
from engine.types import End, GameMode, InvalidMoveError
from strategies.random_strategy import RandomStrategy


# ── Tile Tests ──────────────────────────────────────────────────────────


class TestTile:
    def test_canonical_ordering(self):
        t = Tile(3, 7)
        assert t.high == 7
        assert t.low == 3

    def test_canonical_ordering_reversed(self):
        t = Tile(7, 3)
        assert t.high == 7
        assert t.low == 3

    def test_equality(self):
        assert Tile(3, 7) == Tile(7, 3)

    def test_hash_equality(self):
        assert hash(Tile(3, 7)) == hash(Tile(7, 3))

    def test_pip_sum(self):
        assert Tile(3, 7).pip_sum == 10
        assert Tile(0, 0).pip_sum == 0
        assert Tile(9, 9).pip_sum == 18

    def test_is_double(self):
        assert Tile(5, 5).is_double is True
        assert Tile(3, 7).is_double is False

    def test_has_value(self):
        t = Tile(3, 7)
        assert t.has_value(3) is True
        assert t.has_value(7) is True
        assert t.has_value(5) is False

    def test_other_value(self):
        t = Tile(3, 7)
        assert t.other_value(3) == 7
        assert t.other_value(7) == 3

    def test_other_value_double(self):
        t = Tile(5, 5)
        assert t.other_value(5) == 5

    def test_other_value_invalid(self):
        t = Tile(3, 7)
        with pytest.raises(ValueError):
            t.other_value(5)

    def test_repr(self):
        t = Tile(3, 7)
        assert repr(t) == "[3|7]"

    def test_full_set_count(self, full_tile_set):
        assert len(full_tile_set) == 55

    def test_full_set_unique(self, full_tile_set):
        assert len(set(full_tile_set)) == 55

    def test_full_set_contains_all_doubles(self, full_tile_set):
        doubles = [t for t in full_tile_set if t.is_double]
        assert len(doubles) == 10  # 0-0 through 9-9

    def test_full_set_pip_range(self, full_tile_set):
        for t in full_tile_set:
            assert 0 <= t.low <= t.high <= 9


# ── Board Tests ─────────────────────────────────────────────────────────


class TestBoard:
    def test_empty_board(self):
        b = Board()
        assert b.is_empty is True
        assert b.ends is None
        assert b.played_tiles == frozenset()

    def test_first_tile_normal(self):
        b = Board()
        t = Tile(3, 7)
        b.play(t, "left")
        assert b.is_empty is False
        assert b.ends == (3, 7)
        assert t in b.played_tiles

    def test_first_tile_double(self):
        b = Board()
        t = Tile(5, 5)
        b.play(t, "left")
        assert b.ends == (5, 5)

    def test_play_on_left(self):
        b = Board()
        b.play(Tile(3, 7), "left")  # ends: 3, 7
        b.play(Tile(1, 3), "left")  # left becomes 1
        assert b.ends == (1, 7)

    def test_play_on_right(self):
        b = Board()
        b.play(Tile(3, 7), "left")  # ends: 3, 7
        b.play(Tile(7, 9), "right")  # right becomes 9
        assert b.ends == (3, 9)

    def test_chain_building(self):
        b = Board()
        b.play(Tile(3, 7), "left")  # 3-7
        b.play(Tile(1, 3), "left")  # 1-3-7
        b.play(Tile(7, 9), "right")  # 1-3-7-9
        assert b.ends == (1, 9)
        assert len(b.played_tiles) == 3

    def test_invalid_play_left(self):
        b = Board()
        b.play(Tile(3, 7), "left")
        with pytest.raises(InvalidMoveError):
            b.play(Tile(8, 9), "left")  # neither matches 3

    def test_invalid_play_right(self):
        b = Board()
        b.play(Tile(3, 7), "left")
        with pytest.raises(InvalidMoveError):
            b.play(Tile(1, 2), "right")  # neither matches 7

    def test_can_play(self):
        b = Board()
        b.play(Tile(3, 7), "left")
        assert b.can_play(Tile(3, 5)) is True
        assert b.can_play(Tile(7, 9)) is True
        assert b.can_play(Tile(1, 2)) is False

    def test_can_play_empty_board(self):
        b = Board()
        assert b.can_play(Tile(3, 7)) is True

    def test_valid_placements_both_ends(self):
        b = Board()
        b.play(Tile(3, 7), "left")  # ends: 3, 7
        # Tile matching both ends
        t = Tile(3, 7)
        placements = b.valid_placements(t)
        assert "left" in placements
        assert "right" in placements

    def test_valid_placements_same_ends(self):
        b = Board()
        b.play(Tile(5, 5), "left")  # ends: 5, 5
        t = Tile(5, 3)
        placements = b.valid_placements(t)
        # Both ends are 5, but tile only generates one placement
        # because playing on either end is identical
        assert len(placements) == 1

    def test_valid_placements_double_on_same_ends(self):
        b = Board()
        b.play(Tile(5, 5), "left")  # ends: 5, 5
        t = Tile(5, 5)
        # A double-5 on a board with both ends 5 should have two placements
        placements = b.valid_placements(t)
        assert len(placements) == 2

    def test_valid_placements_empty_board(self):
        b = Board()
        t = Tile(3, 7)
        placements = b.valid_placements(t)
        assert placements == ["left"]


# ── Game Init Tests ─────────────────────────────────────────────────────


class TestGameInit:
    def test_dealing_2p(self, make_game):
        game = make_game(GameMode.FFA_2P)
        for p in game._players:
            assert p.tile_count == 10
        # All tiles unique across hands
        all_tiles = []
        for p in game._players:
            all_tiles.extend(p.hand)
        assert len(set(all_tiles)) == 20

    def test_dealing_3p(self, make_game):
        game = make_game(GameMode.FFA_3P)
        for p in game._players:
            assert p.tile_count == 10
        all_tiles = []
        for p in game._players:
            all_tiles.extend(p.hand)
        assert len(set(all_tiles)) == 30

    def test_dealing_4p(self, make_game):
        game = make_game(GameMode.PAIRS_4P)
        for p in game._players:
            assert p.tile_count == 10
        all_tiles = []
        for p in game._players:
            all_tiles.extend(p.hand)
        assert len(set(all_tiles)) == 40

    def test_no_duplicate_tiles(self, make_game):
        game = make_game(GameMode.PAIRS_4P)
        all_tiles = []
        for p in game._players:
            all_tiles.extend(p.hand)
        assert len(all_tiles) == len(set(all_tiles))

    def test_first_player_by_highest_double(self):
        rng = random.Random(1)
        players = [
            Player(seat=0, strategy=RandomStrategy(rng=rng)),
            Player(seat=1, strategy=RandomStrategy(rng=rng)),
        ]
        # Manually set hands to control first player
        players[0].hand = [Tile(1, 2), Tile(3, 3)]  # has 3-3
        players[1].hand = [Tile(5, 5), Tile(1, 4)]  # has 5-5 (higher)

        game = Game.__new__(Game)
        game._players = players
        game._game_mode = GameMode.FFA_2P
        game._rng = rng
        game._board = Board()
        game._move_history = []
        game._consecutive_passes = 0
        game._game_over = False
        game._is_first_move = True

        first = game._determine_first_player()
        assert first == 1  # seat 1 has 5-5

    def test_first_player_no_doubles_fallback(self):
        rng = random.Random(1)
        players = [
            Player(seat=0, strategy=RandomStrategy(rng=rng)),
            Player(seat=1, strategy=RandomStrategy(rng=rng)),
        ]
        # No doubles in either hand
        players[0].hand = [Tile(1, 2), Tile(3, 4)]  # max pip_sum = 7
        players[1].hand = [Tile(7, 8), Tile(1, 3)]  # max pip_sum = 15

        game = Game.__new__(Game)
        game._players = players
        game._game_mode = GameMode.FFA_2P
        game._rng = rng
        game._board = Board()
        game._move_history = []
        game._consecutive_passes = 0
        game._game_over = False
        game._is_first_move = True

        first = game._determine_first_player()
        assert first == 1  # seat 1 has [7|8] = 15

    def test_invalid_player_count(self):
        rng = random.Random(1)
        players = [Player(seat=0, strategy=RandomStrategy(rng=rng))]
        with pytest.raises(ValueError, match="requires 2 players"):
            Game(players=players, game_mode=GameMode.FFA_2P, rng=rng)

    def test_invalid_player_count_4p(self):
        rng = random.Random(1)
        players = [
            Player(seat=i, strategy=RandomStrategy(rng=rng)) for i in range(3)
        ]
        with pytest.raises(ValueError, match="requires 4 players"):
            Game(players=players, game_mode=GameMode.PAIRS_4P, rng=rng)


# ── Game Loop Tests ─────────────────────────────────────────────────────


class TestGameLoop:
    def test_win_by_empty_hand(self, make_game):
        game = make_game(GameMode.FFA_2P)
        result = game.play_round()
        assert not result.is_stalemate or result.winner_seats
        # At least one player should have finished or stalemate
        assert isinstance(result, RoundResult)

    def test_game_terminates_2p(self, make_game):
        for seed in range(50):
            game = make_game(GameMode.FFA_2P, seed=seed)
            result = game.play_round()
            assert isinstance(result, RoundResult)

    def test_game_terminates_3p(self, make_game):
        for seed in range(50):
            game = make_game(GameMode.FFA_3P, seed=seed)
            result = game.play_round()
            assert isinstance(result, RoundResult)

    def test_game_terminates_4p(self, make_game):
        for seed in range(50):
            game = make_game(GameMode.PAIRS_4P, seed=seed)
            result = game.play_round()
            assert isinstance(result, RoundResult)

    def test_stalemate_detection(self):
        """Force a stalemate: all players can only pass."""
        rng = random.Random(1)
        players = [
            Player(seat=0, strategy=RandomStrategy(rng=rng)),
            Player(seat=1, strategy=RandomStrategy(rng=rng)),
        ]

        game = Game.__new__(Game)
        game._players = players
        game._game_mode = GameMode.FFA_2P
        game._rng = rng
        game._board = Board()
        game._move_history = []
        game._consecutive_passes = 0
        game._game_over = False
        game._is_first_move = False

        # Set up board with end values 0, 1
        game._board.play(Tile(0, 1), "left")

        # Give players tiles that can't match
        players[0].hand = [Tile(5, 6)]
        players[1].hand = [Tile(7, 8)]

        game._current_index = 0
        result = game.play_round()
        assert result.is_stalemate is True

    def test_stalemate_winner_by_pip_sum(self):
        """In stalemate, lower pip sum wins."""
        rng = random.Random(1)
        players = [
            Player(seat=0, strategy=RandomStrategy(rng=rng)),
            Player(seat=1, strategy=RandomStrategy(rng=rng)),
        ]

        game = Game.__new__(Game)
        game._players = players
        game._game_mode = GameMode.FFA_2P
        game._rng = rng
        game._board = Board()
        game._move_history = []
        game._consecutive_passes = 0
        game._game_over = False
        game._is_first_move = False

        game._board.play(Tile(0, 1), "left")
        players[0].hand = [Tile(5, 6)]  # pip sum = 11
        players[1].hand = [Tile(7, 8)]  # pip sum = 15

        game._current_index = 0
        result = game.play_round()
        assert result.is_stalemate is True
        assert result.winner_seats == [0]

    def test_stalemate_tie(self):
        """Equal pip sums = tie (multiple winners)."""
        rng = random.Random(1)
        players = [
            Player(seat=0, strategy=RandomStrategy(rng=rng)),
            Player(seat=1, strategy=RandomStrategy(rng=rng)),
        ]

        game = Game.__new__(Game)
        game._players = players
        game._game_mode = GameMode.FFA_2P
        game._rng = rng
        game._board = Board()
        game._move_history = []
        game._consecutive_passes = 0
        game._game_over = False
        game._is_first_move = False

        game._board.play(Tile(0, 1), "left")
        players[0].hand = [Tile(5, 6)]  # pip sum = 11
        players[1].hand = [Tile(4, 7)]  # pip sum = 11

        game._current_index = 0
        result = game.play_round()
        assert result.is_stalemate is True
        assert result.winner_seats == [0, 1]

    def test_consecutive_pass_reset(self):
        """Playing a tile resets the consecutive pass counter."""
        rng = random.Random(1)
        players = [
            Player(seat=0, strategy=RandomStrategy(rng=rng)),
            Player(seat=1, strategy=RandomStrategy(rng=rng)),
        ]

        game = Game.__new__(Game)
        game._players = players
        game._game_mode = GameMode.FFA_2P
        game._rng = rng
        game._board = Board()
        game._move_history = []
        game._consecutive_passes = 0
        game._game_over = False
        game._is_first_move = True

        # Player 0 plays first tile
        players[0].hand = [Tile(3, 7), Tile(5, 6)]
        players[1].hand = [Tile(7, 8)]

        game._current_index = 0

        # Step: player 0 plays [3|7]
        seat, action, done = game.step()
        assert game._consecutive_passes == 0

    def test_turn_order(self, make_game):
        game = make_game(GameMode.FFA_2P)
        first_seat = game.current_player.seat
        seat, _, _ = game.step()
        assert seat == first_seat
        # Next player is different
        next_seat = game.current_player.seat
        assert next_seat != first_seat

    def test_move_history(self, make_game):
        game = make_game(GameMode.FFA_2P)
        game.play_round()
        assert len(game._move_history) > 0
        for seat, action in game._move_history:
            assert 0 <= seat <= 1
            assert isinstance(action, Action)


# ── Valid Actions Tests ─────────────────────────────────────────────────


class TestValidActions:
    def test_first_move_all_tiles_valid(self):
        rng = random.Random(1)
        p = Player(seat=0, strategy=RandomStrategy(rng=rng))
        p.hand = [Tile(1, 2), Tile(3, 4), Tile(5, 6)]

        game = Game.__new__(Game)
        game._players = [p, Player(seat=1, strategy=RandomStrategy(rng=rng))]
        game._game_mode = GameMode.FFA_2P
        game._board = Board()
        game._is_first_move = True

        actions = game._compute_valid_actions(p)
        assert len(actions) == 3
        tiles_in_actions = {a.tile for a in actions}
        assert tiles_in_actions == set(p.hand)

    def test_normal_play(self):
        rng = random.Random(1)
        p = Player(seat=0, strategy=RandomStrategy(rng=rng))
        p.hand = [Tile(3, 5), Tile(8, 9)]

        game = Game.__new__(Game)
        game._players = [p]
        game._game_mode = GameMode.FFA_2P
        game._board = Board()
        game._board.play(Tile(3, 7), "left")  # ends: 3, 7
        game._is_first_move = False

        actions = game._compute_valid_actions(p)
        play_actions = [a for a in actions if not a.is_pass]
        assert len(play_actions) == 1
        assert play_actions[0].tile == Tile(3, 5)
        assert play_actions[0].end == "left"

    def test_both_ends_tile_generates_two_actions(self):
        rng = random.Random(1)
        p = Player(seat=0, strategy=RandomStrategy(rng=rng))
        p.hand = [Tile(3, 7)]

        game = Game.__new__(Game)
        game._players = [p]
        game._game_mode = GameMode.FFA_2P
        game._board = Board()
        game._board.play(Tile(3, 7), "left")  # ends: 3, 7
        game._is_first_move = False

        actions = game._compute_valid_actions(p)
        assert len(actions) == 2
        ends = {a.end for a in actions}
        assert ends == {"left", "right"}

    def test_forced_pass(self):
        rng = random.Random(1)
        p = Player(seat=0, strategy=RandomStrategy(rng=rng))
        p.hand = [Tile(8, 9)]

        game = Game.__new__(Game)
        game._players = [p]
        game._game_mode = GameMode.FFA_2P
        game._board = Board()
        game._board.play(Tile(3, 7), "left")  # ends: 3, 7
        game._is_first_move = False

        actions = game._compute_valid_actions(p)
        assert len(actions) == 1
        assert actions[0].is_pass is True


# ── Hidden Info Tests ───────────────────────────────────────────────────


class TestHiddenInfo:
    def test_game_state_only_contains_own_hand(self, make_game):
        game = make_game(GameMode.FFA_2P)
        player = game.current_player
        state = game._build_game_state(player)

        # state.hand should match current player's hand
        assert set(state.hand) == set(player.hand)

        # State should not expose other players' hands
        other = [p for p in game._players if p.seat != player.seat][0]
        for tile in other.hand:
            if tile not in player.hand:
                assert tile not in state.hand

    def test_opponent_counts_correct(self, make_game):
        game = make_game(GameMode.FFA_2P)
        player = game.current_player
        state = game._build_game_state(player)
        others = [p for p in game._players if p.seat != player.seat]
        assert state.opponent_tile_counts == tuple(p.tile_count for p in others)

    def test_opponent_counts_3p(self, make_game):
        game = make_game(GameMode.FFA_3P)
        player = game.current_player
        state = game._build_game_state(player)
        others = [p for p in game._players if p.seat != player.seat]
        assert len(state.opponent_tile_counts) == 2
        assert state.opponent_tile_counts == tuple(p.tile_count for p in others)

    def test_game_state_no_hand_leaks(self, make_game):
        """GameState frozen dataclass should only have known fields."""
        game = make_game(GameMode.FFA_2P)
        player = game.current_player
        state = game._build_game_state(player)
        # Verify there's no attribute that could leak other hands
        field_names = {f.name for f in state.__dataclass_fields__.values()}
        dangerous = {"other_hands", "all_hands", "deck", "undealt"}
        assert field_names & dangerous == set()


# ── Pairs Tests ─────────────────────────────────────────────────────────


class TestPairs:
    def test_team_assignment(self, make_game):
        game = make_game(GameMode.PAIRS_4P)
        player_0 = game._players[0]
        state = game._build_game_state(player_0)
        assert state.teammate_seat == 2

        player_1 = game._players[1]
        state = game._build_game_state(player_1)
        assert state.teammate_seat == 3

    def test_pairs_no_teammate_in_ffa(self, make_game):
        game = make_game(GameMode.FFA_2P)
        player = game.current_player
        state = game._build_game_state(player)
        assert state.teammate_seat is None

    def test_pairs_game_completes(self, make_game):
        game = make_game(GameMode.PAIRS_4P)
        result = game.play_round()
        assert isinstance(result, RoundResult)
        if not result.is_stalemate:
            # Winner seats should be a team (both members)
            assert len(result.winner_seats) == 2
            s0, s1 = result.winner_seats
            assert abs(s0 - s1) == 2  # teammates are 2 apart

    def test_pairs_points_calculation(self):
        """Points = opposing team's remaining pips in pairs."""
        rng = random.Random(1)
        players = [
            Player(seat=i, strategy=RandomStrategy(rng=rng)) for i in range(4)
        ]

        game = Game.__new__(Game)
        game._players = players
        game._game_mode = GameMode.PAIRS_4P
        game._rng = rng
        game._board = Board()
        game._move_history = []
        game._consecutive_passes = 0
        game._game_over = False
        game._is_first_move = True

        # Player 0 wins by emptying hand
        players[0].hand = [Tile(3, 7)]
        players[1].hand = [Tile(1, 2), Tile(4, 5)]  # pip sum = 12
        players[2].hand = [Tile(0, 1)]  # teammate of 0
        players[3].hand = [Tile(6, 8), Tile(2, 3)]  # pip sum = 19

        game._current_index = 0

        # Player 0 plays their only tile
        seat, action, done = game.step()
        assert done is True

        result = game._build_result()
        assert result.winner_seats == [0, 2]
        # Points = opposing team (seats 1,3) remaining pips
        expected_points = players[1].pip_sum + players[3].pip_sum
        assert result.points == expected_points

    def test_pairs_stalemate_team_win(self):
        """In pairs stalemate, player with lowest individual pip sum wins for their team."""
        rng = random.Random(1)
        players = [
            Player(seat=i, strategy=RandomStrategy(rng=rng)) for i in range(4)
        ]

        game = Game.__new__(Game)
        game._players = players
        game._game_mode = GameMode.PAIRS_4P
        game._rng = rng
        game._board = Board()
        game._move_history = []
        game._consecutive_passes = 0
        game._game_over = False
        game._is_first_move = False

        game._board.play(Tile(0, 1), "left")

        # Seat 0 (Team A): pip_sum = 5 — LOWEST individual
        players[0].hand = [Tile(2, 3)]
        # Seat 2 (Team A): pip_sum = 17
        players[2].hand = [Tile(8, 9)]
        # Seat 1 (Team B): pip_sum = 7
        players[1].hand = [Tile(3, 4)]
        # Seat 3 (Team B): pip_sum = 6
        players[3].hand = [Tile(2, 4)]
        # Team A total = 22, Team B total = 13
        # But seat 0 has the lowest individual (5), so Team A wins

        game._current_index = 0
        result = game.play_round()
        assert result.is_stalemate is True
        assert result.winner_seats == [0, 2]  # Team A wins despite higher team total


# ── Bulk Smoke Tests ────────────────────────────────────────────────────


class TestBulkSmoke:
    def test_100_games_2p(self):
        for seed in range(100):
            rng = random.Random(seed)
            players = [
                Player(seat=i, strategy=RandomStrategy(rng=random.Random(seed + i)))
                for i in range(2)
            ]
            game = Game(players=players, game_mode=GameMode.FFA_2P, rng=rng)
            result = game.play_round()
            assert isinstance(result, RoundResult)
            assert len(result.move_history) > 0

    def test_100_games_3p(self):
        for seed in range(100):
            rng = random.Random(seed)
            players = [
                Player(seat=i, strategy=RandomStrategy(rng=random.Random(seed + i)))
                for i in range(3)
            ]
            game = Game(players=players, game_mode=GameMode.FFA_3P, rng=rng)
            result = game.play_round()
            assert isinstance(result, RoundResult)

    def test_100_games_4p(self):
        for seed in range(100):
            rng = random.Random(seed)
            players = [
                Player(seat=i, strategy=RandomStrategy(rng=random.Random(seed + i)))
                for i in range(4)
            ]
            game = Game(players=players, game_mode=GameMode.PAIRS_4P, rng=rng)
            result = game.play_round()
            assert isinstance(result, RoundResult)


# ── Reproducibility Tests ──────────────────────────────────────────────


class TestReproducibility:
    def test_same_seed_same_result(self):
        def run_game(seed: int) -> list[tuple[int, Action]]:
            rng = random.Random(seed)
            players = [
                Player(
                    seat=i,
                    strategy=RandomStrategy(rng=random.Random(seed + i)),
                )
                for i in range(2)
            ]
            game = Game(players=players, game_mode=GameMode.FFA_2P, rng=rng)
            result = game.play_round()
            return result.move_history

        history1 = run_game(12345)
        history2 = run_game(12345)
        assert len(history1) == len(history2)
        for (s1, a1), (s2, a2) in zip(history1, history2):
            assert s1 == s2
            assert a1 == a2

    def test_different_seed_different_result(self):
        def run_game(seed: int) -> list[tuple[int, Action]]:
            rng = random.Random(seed)
            players = [
                Player(
                    seat=i,
                    strategy=RandomStrategy(rng=random.Random(seed + i)),
                )
                for i in range(2)
            ]
            game = Game(players=players, game_mode=GameMode.FFA_2P, rng=rng)
            result = game.play_round()
            return result.move_history

        history1 = run_game(111)
        history2 = run_game(222)
        # Very unlikely to be identical
        if len(history1) == len(history2):
            differs = any(
                s1 != s2 or a1 != a2
                for (s1, a1), (s2, a2) in zip(history1, history2)
            )
            assert differs
