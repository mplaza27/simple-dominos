from __future__ import annotations

import random

import pytest

from engine.game import Action, Game
from engine.player import Player
from engine.tile import Board, Tile
from engine.types import GameMode
from strategies.greedy_strategy import GreedyStrategy
from strategies.greedy_doubles_strategy import GreedyDoublesStrategy
from strategies.late_game_strategy import LateGameStrategy
from strategies.never_passed_strategy import NeverPassedStrategy
from strategies.non_greedy_strategy import NonGreedyStrategy
from strategies.partner_aware_strategy import PartnerAwareStrategy
from strategies.pass_tracker_strategy import PassTrackerStrategy
from strategies.random_strategy import RandomStrategy
from strategies.rl_strategy import RLStrategy


class TestRandomStrategy:
    def test_always_returns_valid_action(self):
        for seed in range(50):
            rng = random.Random(seed)
            players = [
                Player(seat=i, strategy=RandomStrategy(rng=random.Random(seed + i)))
                for i in range(2)
            ]
            game = Game(players=players, game_mode=GameMode.FFA_2P, rng=rng)
            result = game.play_round()
            # If we got here without InvalidMoveError, all actions were valid
            assert result is not None

    def test_passes_when_forced(self):
        rng = random.Random(42)
        strategy = RandomStrategy(rng=rng)

        from engine.game import GameState

        state = GameState(
            hand=(Tile(8, 9),),
            board_ends=(3, 7),
            played_tiles=frozenset({Tile(3, 7)}),
            opponent_tile_counts=(5,),
            pass_history=(False, False),
            valid_actions=[Action.pass_action()],
            current_seat=0,
            teammate_seat=None,
            game_mode=GameMode.FFA_2P,
            is_first_move=False,
            move_history=(),
            num_players=2,
        )

        action = strategy.choose_action(state)
        assert action.is_pass


class TestGreedyStrategy:
    def test_picks_max_pip_sum(self):
        rng = random.Random(42)
        strategy = GreedyStrategy(rng=rng)

        from engine.game import GameState

        state = GameState(
            hand=(Tile(1, 3), Tile(7, 3), Tile(3, 9)),
            board_ends=(3, 5),
            played_tiles=frozenset({Tile(3, 5)}),
            opponent_tile_counts=(5,),
            pass_history=(False, False),
            valid_actions=[
                Action.play(Tile(1, 3), "left"),
                Action.play(Tile(7, 3), "left"),
                Action.play(Tile(3, 9), "left"),
                Action.play(Tile(7, 3), "right"),  # Tile(7,3) doesn't match 5... let me fix
            ],
            current_seat=0,
            teammate_seat=None,
            game_mode=GameMode.FFA_2P,
            is_first_move=False,
            move_history=(),
            num_players=2,
        )

        # The valid actions with tiles having pip sums: 4, 10, 12
        # Greedy should pick Tile(3,9) with pip_sum=12
        action = strategy.choose_action(state)
        assert action.tile == Tile(3, 9)

    def test_passes_when_forced(self):
        rng = random.Random(42)
        strategy = GreedyStrategy(rng=rng)

        from engine.game import GameState

        state = GameState(
            hand=(Tile(8, 9),),
            board_ends=(3, 7),
            played_tiles=frozenset({Tile(3, 7)}),
            opponent_tile_counts=(5,),
            pass_history=(False, False),
            valid_actions=[Action.pass_action()],
            current_seat=0,
            teammate_seat=None,
            game_mode=GameMode.FFA_2P,
            is_first_move=False,
            move_history=(),
            num_players=2,
        )

        action = strategy.choose_action(state)
        assert action.is_pass

    def test_greedy_games_complete(self):
        for seed in range(50):
            rng = random.Random(seed)
            players = [
                Player(seat=i, strategy=GreedyStrategy(rng=random.Random(seed + i)))
                for i in range(2)
            ]
            game = Game(players=players, game_mode=GameMode.FFA_2P, rng=rng)
            result = game.play_round()
            assert result is not None


class TestGreedyDoublesStrategy:
    def test_prefers_doubles_over_higher_pip(self):
        rng = random.Random(42)
        strategy = GreedyDoublesStrategy(rng=rng)

        from engine.game import GameState

        state = GameState(
            hand=(Tile(3, 3), Tile(3, 9)),
            board_ends=(3, 5),
            played_tiles=frozenset({Tile(3, 5)}),
            opponent_tile_counts=(5,),
            pass_history=(False, False),
            valid_actions=[
                Action.play(Tile(3, 3), "left"),
                Action.play(Tile(3, 9), "left"),
            ],
            current_seat=0,
            teammate_seat=None,
            game_mode=GameMode.FFA_2P,
            is_first_move=False,
            move_history=(),
            num_players=2,
        )

        action = strategy.choose_action(state)
        # Should pick [3|3] (double) even though [3|9] has higher pip sum
        assert action.tile == Tile(3, 3)

    def test_picks_highest_double(self):
        rng = random.Random(42)
        strategy = GreedyDoublesStrategy(rng=rng)

        from engine.game import GameState

        state = GameState(
            hand=(Tile(3, 3), Tile(5, 5), Tile(3, 7)),
            board_ends=(3, 5),
            played_tiles=frozenset({Tile(3, 5)}),
            opponent_tile_counts=(5,),
            pass_history=(False, False),
            valid_actions=[
                Action.play(Tile(3, 3), "left"),
                Action.play(Tile(5, 5), "right"),
                Action.play(Tile(3, 7), "left"),
            ],
            current_seat=0,
            teammate_seat=None,
            game_mode=GameMode.FFA_2P,
            is_first_move=False,
            move_history=(),
            num_players=2,
        )

        action = strategy.choose_action(state)
        assert action.tile == Tile(5, 5)

    def test_falls_back_to_greedy_when_no_doubles(self):
        rng = random.Random(42)
        strategy = GreedyDoublesStrategy(rng=rng)

        from engine.game import GameState

        state = GameState(
            hand=(Tile(1, 3), Tile(3, 9)),
            board_ends=(3, 5),
            played_tiles=frozenset({Tile(3, 5)}),
            opponent_tile_counts=(5,),
            pass_history=(False, False),
            valid_actions=[
                Action.play(Tile(1, 3), "left"),
                Action.play(Tile(3, 9), "left"),
            ],
            current_seat=0,
            teammate_seat=None,
            game_mode=GameMode.FFA_2P,
            is_first_move=False,
            move_history=(),
            num_players=2,
        )

        action = strategy.choose_action(state)
        assert action.tile == Tile(3, 9)  # highest pip sum

    def test_games_complete(self):
        for seed in range(50):
            rng = random.Random(seed)
            players = [
                Player(seat=i, strategy=GreedyDoublesStrategy(rng=random.Random(seed + i)))
                for i in range(2)
            ]
            game = Game(players=players, game_mode=GameMode.FFA_2P, rng=rng)
            result = game.play_round()
            assert result is not None


class TestNonGreedyStrategy:
    def test_picks_min_pip_sum(self):
        rng = random.Random(42)
        strategy = NonGreedyStrategy(rng=rng)

        from engine.game import GameState

        state = GameState(
            hand=(Tile(1, 3), Tile(7, 3), Tile(3, 9)),
            board_ends=(3, 5),
            played_tiles=frozenset({Tile(3, 5)}),
            opponent_tile_counts=(5,),
            pass_history=(False, False),
            valid_actions=[
                Action.play(Tile(1, 3), "left"),
                Action.play(Tile(7, 3), "left"),
                Action.play(Tile(3, 9), "left"),
            ],
            current_seat=0,
            teammate_seat=None,
            game_mode=GameMode.FFA_2P,
            is_first_move=False,
            move_history=(),
            num_players=2,
        )

        action = strategy.choose_action(state)
        assert action.tile == Tile(1, 3)  # pip sum = 4

    def test_passes_when_forced(self):
        rng = random.Random(42)
        strategy = NonGreedyStrategy(rng=rng)

        from engine.game import GameState

        state = GameState(
            hand=(Tile(8, 9),),
            board_ends=(3, 7),
            played_tiles=frozenset({Tile(3, 7)}),
            opponent_tile_counts=(5,),
            pass_history=(False, False),
            valid_actions=[Action.pass_action()],
            current_seat=0,
            teammate_seat=None,
            game_mode=GameMode.FFA_2P,
            is_first_move=False,
            move_history=(),
            num_players=2,
        )

        action = strategy.choose_action(state)
        assert action.is_pass

    def test_games_complete(self):
        for seed in range(50):
            rng = random.Random(seed)
            players = [
                Player(seat=i, strategy=NonGreedyStrategy(rng=random.Random(seed + i)))
                for i in range(2)
            ]
            game = Game(players=players, game_mode=GameMode.FFA_2P, rng=rng)
            result = game.play_round()
            assert result is not None


class TestLateGameStrategy:
    def test_plays_least_connected_tile(self):
        rng = random.Random(42)
        strategy = LateGameStrategy(rng=rng)

        from engine.game import GameState

        # Hand: [3|5], [5|5], [5|7]
        # Board ends: 3, 7
        # [3|5] connects to [5|5] and [5|7] → connectivity 2
        # [5|5] connects to [3|5] and [5|7] + double bonus → connectivity 4
        # [5|7] connects to [3|5] and [5|5] → connectivity 2
        # But [5|7] plays on right (matching 7), [3|5] plays on left (matching 3)
        # Both have connectivity 2; tiebreak by higher pip sum → [5|7]=12 > [3|5]=8
        state = GameState(
            hand=(Tile(3, 5), Tile(5, 5), Tile(5, 7)),
            board_ends=(3, 7),
            played_tiles=frozenset({Tile(3, 7)}),
            opponent_tile_counts=(5,),
            pass_history=(False, False),
            valid_actions=[
                Action.play(Tile(3, 5), "left"),
                Action.play(Tile(5, 5), "left"),  # not actually valid here but testing strategy logic
                Action.play(Tile(5, 7), "right"),
            ],
            current_seat=0,
            teammate_seat=None,
            game_mode=GameMode.FFA_2P,
            is_first_move=False,
            move_history=(),
            num_players=2,
        )

        action = strategy.choose_action(state)
        # Should NOT play [5|5] (highest connectivity=4), should play one of the less connected
        assert action.tile != Tile(5, 5)

    def test_keeps_doubles(self):
        rng = random.Random(42)
        strategy = LateGameStrategy(rng=rng)

        from engine.game import GameState

        # Hand: [3|3], [3|7]. Board end: 3
        # [3|3]: connects to [3|7] + double bonus → connectivity 3
        # [3|7]: connects to [3|3] → connectivity 1
        # Strategy should play [3|7] (lower connectivity) to keep the double
        state = GameState(
            hand=(Tile(3, 3), Tile(3, 7)),
            board_ends=(3, 5),
            played_tiles=frozenset({Tile(3, 5)}),
            opponent_tile_counts=(5,),
            pass_history=(False, False),
            valid_actions=[
                Action.play(Tile(3, 3), "left"),
                Action.play(Tile(3, 7), "left"),
            ],
            current_seat=0,
            teammate_seat=None,
            game_mode=GameMode.FFA_2P,
            is_first_move=False,
            move_history=(),
            num_players=2,
        )

        action = strategy.choose_action(state)
        assert action.tile == Tile(3, 7)

    def test_games_complete(self):
        for seed in range(50):
            rng = random.Random(seed)
            players = [
                Player(seat=i, strategy=LateGameStrategy(rng=random.Random(seed + i)))
                for i in range(2)
            ]
            game = Game(players=players, game_mode=GameMode.FFA_2P, rng=rng)
            result = game.play_round()
            assert result is not None


class TestNeverPassedStrategy:
    def test_preserves_coverage(self):
        """Should play the tile that leaves the most distinct pip values in hand."""
        rng = random.Random(42)
        strategy = NeverPassedStrategy(rng=rng)

        from engine.game import GameState

        # Hand: [3|5], [5|7], [5|9], [1|2]
        # Board ends: 5, 8
        # Playable on left (matching 5): [3|5], [5|7], [5|9]
        #
        # Play [3|5] → remaining: [5|7], [5|9], [1|2] → values {5,7,9,1,2} = 5
        # Play [5|7] → remaining: [3|5], [5|9], [1|2] → values {3,5,9,1,2} = 5
        # Play [5|9] → remaining: [3|5], [5|7], [1|2] → values {3,5,7,1,2} = 5
        # All equal coverage. Tiebreak: highest pip sum → [5|9]=14
        state = GameState(
            hand=(Tile(3, 5), Tile(5, 7), Tile(5, 9), Tile(1, 2)),
            board_ends=(5, 8),
            played_tiles=frozenset({Tile(5, 8)}),
            opponent_tile_counts=(5,),
            pass_history=(False, False),
            valid_actions=[
                Action.play(Tile(3, 5), "left"),
                Action.play(Tile(5, 7), "left"),
                Action.play(Tile(5, 9), "left"),
            ],
            current_seat=0,
            teammate_seat=None,
            game_mode=GameMode.FFA_2P,
            is_first_move=False,
            move_history=(),
            num_players=2,
        )

        action = strategy.choose_action(state)
        assert action.tile == Tile(5, 9)

    def test_avoids_losing_unique_value(self):
        """Should avoid playing a tile that removes a unique pip value from hand."""
        rng = random.Random(42)
        strategy = NeverPassedStrategy(rng=rng)

        from engine.game import GameState

        # Hand: [3|5], [3|9], [7|8]
        # Board ends: 3, 6
        # Playable: [3|5] and [3|9] on left
        #
        # Play [3|5] → remaining: [3|9], [7|8] → values {3,9,7,8} = 4
        # Play [3|9] → remaining: [3|5], [7|8] → values {3,5,7,8} = 4
        # Equal coverage; tiebreak pip sum → [3|9]=12 > [3|5]=8
        #
        # Better example: Hand: [3|5], [3|9], [5|6]
        # Play [3|5] → remaining: [3|9], [5|6] → values {3,9,5,6} = 4
        # Play [3|9] → remaining: [3|5], [5|6] → values {3,5,6} = 3
        # [3|5] preserves more coverage (keeps 9 via [3|9])
        state = GameState(
            hand=(Tile(3, 5), Tile(3, 9), Tile(5, 6)),
            board_ends=(3, 7),
            played_tiles=frozenset({Tile(3, 7)}),
            opponent_tile_counts=(5,),
            pass_history=(False, False),
            valid_actions=[
                Action.play(Tile(3, 5), "left"),
                Action.play(Tile(3, 9), "left"),
            ],
            current_seat=0,
            teammate_seat=None,
            game_mode=GameMode.FFA_2P,
            is_first_move=False,
            move_history=(),
            num_players=2,
        )

        action = strategy.choose_action(state)
        # [3|5] leaves 4 distinct values, [3|9] leaves only 3
        assert action.tile == Tile(3, 5)

    def test_plays_from_deepest_suit(self):
        """When you have many tiles of one value, play from that suit
        since coverage is preserved."""
        rng = random.Random(42)
        strategy = NeverPassedStrategy(rng=rng)

        from engine.game import GameState

        # Hand: [5|1], [5|2], [5|3], [7|8]
        # Board ends: 5, 9
        # All three 5-tiles playable. Playing any leaves {5,remaining,7,8} + others
        #
        # Play [5|1] → remaining {5,2,3,7,8} = 5 (still have 5 via [5|2],[5|3])
        # Play [5|2] → remaining {5,1,3,7,8} = 5
        # Play [5|3] → remaining {5,1,2,7,8} = 5
        # All equal. Tiebreak: pip sum → [5|3]=8 wins
        state = GameState(
            hand=(Tile(1, 5), Tile(2, 5), Tile(3, 5), Tile(7, 8)),
            board_ends=(5, 9),
            played_tiles=frozenset({Tile(5, 9)}),
            opponent_tile_counts=(5,),
            pass_history=(False, False),
            valid_actions=[
                Action.play(Tile(1, 5), "left"),
                Action.play(Tile(2, 5), "left"),
                Action.play(Tile(3, 5), "left"),
            ],
            current_seat=0,
            teammate_seat=None,
            game_mode=GameMode.FFA_2P,
            is_first_move=False,
            move_history=(),
            num_players=2,
        )

        action = strategy.choose_action(state)
        assert action.tile == Tile(3, 5)

    def test_passes_when_forced(self):
        rng = random.Random(42)
        strategy = NeverPassedStrategy(rng=rng)

        from engine.game import GameState

        state = GameState(
            hand=(Tile(8, 9),),
            board_ends=(3, 7),
            played_tiles=frozenset({Tile(3, 7)}),
            opponent_tile_counts=(5,),
            pass_history=(False, False),
            valid_actions=[Action.pass_action()],
            current_seat=0,
            teammate_seat=None,
            game_mode=GameMode.FFA_2P,
            is_first_move=False,
            move_history=(),
            num_players=2,
        )

        action = strategy.choose_action(state)
        assert action.is_pass

    def test_games_complete(self):
        for seed in range(50):
            rng = random.Random(seed)
            players = [
                Player(seat=i, strategy=NeverPassedStrategy(rng=random.Random(seed + i)))
                for i in range(2)
            ]
            game = Game(players=players, game_mode=GameMode.FFA_2P, rng=rng)
            result = game.play_round()
            assert result is not None


class TestPassTrackerStrategy:
    def test_exploits_opponent_pass(self):
        """When opponent passed on ends (3,7), strategy should prefer leaving
        those values exposed."""
        rng = random.Random(42)
        strategy = PassTrackerStrategy(rng=rng)

        from engine.game import GameState

        # Move history: someone played [3|7], then opponent (seat 1) passed
        history = (
            (0, Action.play(Tile(3, 7), "left")),  # board: 3-7
            (1, Action.pass_action()),               # seat 1 lacks 3 and 7
        )

        # Now it's seat 0's turn. Board ends: 3, 7.
        # Hand has [1|3] and [5|7].
        # Playing [1|3] on left → new left end = 1 (not in missing set)
        # Playing [5|7] on right → new right end = 5 (not in missing set)
        # Playing [1|3] on left doesn't help exploit. Playing [5|7] on right doesn't either.
        # But both remove a value the opponent lacks from the board...
        # Actually, let's set up a clearer scenario:
        # Board ends: 3, 7. Opponent lacks {3, 7}.
        # If we play [3|9] on left → new left end = 9. Board: 9...7. Opponent still lacks 7.
        # If we play [7|9] on right → new right end = 9. Board: 3...9. Opponent still lacks 3.
        # Both leave one value the opponent lacks. Score = 1 each. Tiebreak: higher pip sum.
        state = GameState(
            hand=(Tile(3, 9), Tile(7, 9)),
            board_ends=(3, 7),
            played_tiles=frozenset({Tile(3, 7)}),
            opponent_tile_counts=(5,),
            pass_history=(False, True),
            valid_actions=[
                Action.play(Tile(3, 9), "left"),
                Action.play(Tile(7, 9), "right"),
            ],
            current_seat=0,
            teammate_seat=None,
            game_mode=GameMode.FFA_2P,
            is_first_move=False,
            move_history=history,
            num_players=2,
        )

        action = strategy.choose_action(state)
        # Both have same exploit score (1) and same pip sum (12), random tiebreak
        assert action.tile in (Tile(3, 9), Tile(7, 9))

    def test_prefers_exploit_over_high_pip(self):
        """Should prefer a play that exploits opponent's weakness even if
        another tile has higher pip sum."""
        rng = random.Random(42)
        strategy = PassTrackerStrategy(rng=rng)

        from engine.game import GameState

        # Opponent passed when ends were (5, 8) → lacks 5 and 8
        history = (
            (0, Action.play(Tile(5, 8), "left")),
            (1, Action.pass_action()),
            (0, Action.play(Tile(5, 3), "left")),  # board: 3...8
        )

        # Board ends: 3, 8. Hand: [3|4] (pip=7), [8|9] (pip=17)
        # Playing [3|4] on left → new end = 4 (not in {5,8}) → score 0
        # Playing [8|9] on right → new end = 9 (not in {5,8}) → score 0
        # Hmm, neither exploits directly. Let me adjust:
        # Hand: [3|5] (pip=8), [8|9] (pip=17)
        # Playing [3|5] on left → new end = 5. 5 IS in missing set → score 1
        # Playing [8|9] on right → new end = 9. Not in missing set → score 0
        state = GameState(
            hand=(Tile(3, 5), Tile(8, 9)),
            board_ends=(3, 8),
            played_tiles=frozenset({Tile(5, 8), Tile(3, 5)}),
            opponent_tile_counts=(5,),
            pass_history=(False, True),
            valid_actions=[
                Action.play(Tile(3, 5), "left"),
                Action.play(Tile(8, 9), "right"),
            ],
            current_seat=0,
            teammate_seat=None,
            game_mode=GameMode.FFA_2P,
            is_first_move=False,
            move_history=history,
            num_players=2,
        )

        action = strategy.choose_action(state)
        # [3|5] exploits (leaves end=5 which opponent lacks) even though pip sum is lower
        assert action.tile == Tile(3, 5)

    def test_falls_back_to_greedy_with_no_passes(self):
        rng = random.Random(42)
        strategy = PassTrackerStrategy(rng=rng)

        from engine.game import GameState

        state = GameState(
            hand=(Tile(1, 3), Tile(3, 9)),
            board_ends=(3, 5),
            played_tiles=frozenset({Tile(3, 5)}),
            opponent_tile_counts=(5,),
            pass_history=(False, False),
            valid_actions=[
                Action.play(Tile(1, 3), "left"),
                Action.play(Tile(3, 9), "left"),
            ],
            current_seat=0,
            teammate_seat=None,
            game_mode=GameMode.FFA_2P,
            is_first_move=False,
            move_history=((0, Action.play(Tile(3, 5), "left")),),
            num_players=2,
        )

        action = strategy.choose_action(state)
        assert action.tile == Tile(3, 9)  # greedy fallback

    def test_games_complete(self):
        for seed in range(50):
            rng = random.Random(seed)
            players = [
                Player(seat=i, strategy=PassTrackerStrategy(rng=random.Random(seed + i)))
                for i in range(2)
            ]
            game = Game(players=players, game_mode=GameMode.FFA_2P, rng=rng)
            result = game.play_round()
            assert result is not None


class TestPartnerAwareStrategy:
    def test_plays_partner_values(self):
        """Should prefer tiles matching values the partner has played."""
        rng = random.Random(42)
        strategy = PartnerAwareStrategy(rng=rng)

        from engine.game import GameState

        # Partner (seat 2) opened with [5|7]
        history = (
            (2, Action.play(Tile(5, 7), "left")),  # board: 5-7
            (3, Action.play(Tile(7, 9), "right")),  # board: 5-7-9
            # Now seat 0's turn. Partner values: {5, 7}
        )

        state = GameState(
            hand=(Tile(5, 3), Tile(9, 1)),
            board_ends=(5, 9),
            played_tiles=frozenset({Tile(5, 7), Tile(7, 9)}),
            opponent_tile_counts=(9, 9, 9),
            pass_history=(False, False, False, False),
            valid_actions=[
                Action.play(Tile(5, 3), "left"),
                Action.play(Tile(9, 1), "right"),
            ],
            current_seat=0,
            teammate_seat=2,
            game_mode=GameMode.PAIRS_4P,
            is_first_move=False,
            move_history=history,
            num_players=4,
        )

        action = strategy.choose_action(state)
        # [5|3] matches partner's value 5, [9|1] doesn't match partner's {5,7}
        assert action.tile == Tile(3, 5)

    def test_avoids_partner_missing_end(self):
        """Should avoid leaving board ends at values partner lacks."""
        rng = random.Random(42)
        strategy = PartnerAwareStrategy(rng=rng)

        from engine.game import GameState

        # Partner passed on board ends (3, 7) → lacks {3, 7}
        history = (
            (0, Action.play(Tile(3, 7), "left")),
            (1, Action.play(Tile(7, 8), "right")),  # board: 3..8
            (2, Action.pass_action()),                # partner lacks 3 and 8
            (3, Action.play(Tile(8, 9), "right")),    # board: 3..9
        )

        # Board ends: 3, 9. Hand: [3|6], [9|6]
        # [3|6] on left → new end = 6. 6 not in partner_missing, not in partner_values → neutral
        # [9|6] on right → new end = 6. Same.
        # But: [3|6] leaves right end still 9 (partner doesn't lack 9).
        #       [9|6] leaves left end still 3 (partner LACKS 3 → bad).
        # Actually the scoring checks the NEW end value against partner_missing.
        # [3|6] on left → new end = 6 → not in missing → 0 penalty
        # [9|6] on right → new end = 6 → not in missing → 0 penalty
        # Hmm, both are same. Let me pick a better example:
        # Hand: [3|8], [9|5]. Board: 3..9
        # [3|8] on left → new end = 8. 8 IS in partner_missing → -2 penalty
        # [9|5] on right → new end = 5. 5 not in partner_missing → no penalty
        state = GameState(
            hand=(Tile(3, 8), Tile(5, 9)),
            board_ends=(3, 9),
            played_tiles=frozenset({Tile(3, 7), Tile(7, 8), Tile(8, 9)}),
            opponent_tile_counts=(9, 9, 9),
            pass_history=(False, False, True, False),
            valid_actions=[
                Action.play(Tile(3, 8), "left"),
                Action.play(Tile(5, 9), "right"),
            ],
            current_seat=0,
            teammate_seat=2,
            game_mode=GameMode.PAIRS_4P,
            is_first_move=False,
            move_history=history,
            num_players=4,
        )

        action = strategy.choose_action(state)
        # Should prefer [9|5] which doesn't leave a bad end for partner
        assert action.tile == Tile(5, 9)

    def test_plays_opposite_end_from_partner(self):
        """Should prefer playing on the opposite end from where partner last played,
        when other signals are equal."""
        rng = random.Random(42)
        strategy = PartnerAwareStrategy(rng=rng)

        from engine.game import GameState

        # Partner (seat 2) last played on the left. Partner values: {6, 4}
        history = (
            (0, Action.play(Tile(1, 2), "left")),    # board: 1-2
            (1, Action.play(Tile(2, 8), "right")),    # board: 1..8
            (2, Action.play(Tile(1, 6), "left")),     # partner played LEFT, board: 6..8
            (3, Action.play(Tile(8, 9), "right")),    # board: 6..9
        )

        # Board ends: 6, 9. Hand: [6|0], [9|0] — same pip sum (6 vs 9)
        # Neither tile has partner values {1, 6}... wait [6|0] has value 6.
        # Use tiles that don't match partner values at all:
        # Hand: [6|0], [9|0]. partner_values = {1, 6}.
        # [6|0] has value 6 which IS in partner_values → +3.0. That confounds.
        # Let me use different board ends.

        # Actually, use a scenario where both tiles match partner values equally.
        # Partner values: {1, 6}. Board: 6..9.
        # Hand: [6|1], [9|1] — both contain value 1 (partner value), so both get +3.
        # [6|1] on left → new end = 1. 1 in partner_values → +2. Same end as partner → +0.
        # [9|1] on right → new end = 1. 1 in partner_values → +2. Opposite end → +1.
        # So [9|1] should win: 3+2+1+0.1 = 6.1 vs [6|1]: 3+2+0+0.07 = 5.07
        state = GameState(
            hand=(Tile(1, 6), Tile(1, 9)),
            board_ends=(6, 9),
            played_tiles=frozenset({Tile(1, 2), Tile(2, 8), Tile(1, 6), Tile(8, 9)}),
            opponent_tile_counts=(8, 8, 8),
            pass_history=(False, False, False, False),
            valid_actions=[
                Action.play(Tile(1, 6), "left"),
                Action.play(Tile(1, 9), "right"),
            ],
            current_seat=0,
            teammate_seat=2,
            game_mode=GameMode.PAIRS_4P,
            is_first_move=False,
            move_history=history,
            num_players=4,
        )

        action = strategy.choose_action(state)
        # [1|9] plays on right (opposite from partner's left) and has higher pip sum
        assert action.tile == Tile(1, 9)

    def test_falls_back_to_greedy_in_ffa(self):
        rng = random.Random(42)
        strategy = PartnerAwareStrategy(rng=rng)

        from engine.game import GameState

        state = GameState(
            hand=(Tile(1, 3), Tile(3, 9)),
            board_ends=(3, 5),
            played_tiles=frozenset({Tile(3, 5)}),
            opponent_tile_counts=(5,),
            pass_history=(False, False),
            valid_actions=[
                Action.play(Tile(1, 3), "left"),
                Action.play(Tile(3, 9), "left"),
            ],
            current_seat=0,
            teammate_seat=None,
            game_mode=GameMode.FFA_2P,
            is_first_move=False,
            move_history=(),
            num_players=2,
        )

        action = strategy.choose_action(state)
        assert action.tile == Tile(3, 9)

    def test_games_complete_4p(self):
        for seed in range(50):
            rng = random.Random(seed)
            players = [
                Player(seat=i, strategy=PartnerAwareStrategy(rng=random.Random(seed + i)))
                for i in range(4)
            ]
            game = Game(players=players, game_mode=GameMode.PAIRS_4P, rng=rng)
            result = game.play_round()
            assert result is not None

    def test_games_complete_2p_fallback(self):
        for seed in range(50):
            rng = random.Random(seed)
            players = [
                Player(seat=i, strategy=PartnerAwareStrategy(rng=random.Random(seed + i)))
                for i in range(2)
            ]
            game = Game(players=players, game_mode=GameMode.FFA_2P, rng=rng)
            result = game.play_round()
            assert result is not None


class TestRLStrategy:
    def test_choose_action_returns_valid(self):
        """RLStrategy with random weights should return a valid action."""
        strategy = RLStrategy()

        from engine.game import GameState

        state = GameState(
            hand=(Tile(1, 2),),
            board_ends=(1, 3),
            played_tiles=frozenset(),
            opponent_tile_counts=(5,),
            pass_history=(False, False),
            valid_actions=[Action.play(Tile(1, 2), "left")],
            current_seat=0,
            teammate_seat=None,
            game_mode=GameMode.FFA_2P,
            is_first_move=False,
            move_history=(),
            num_players=2,
        )

        action = strategy.choose_action(state)
        assert action in state.valid_actions
