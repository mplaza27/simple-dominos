from __future__ import annotations

import random

import pytest

from engine.game import Game
from engine.player import Player
from engine.tile import Tile, create_full_set
from engine.types import GameMode
from strategies.random_strategy import RandomStrategy


@pytest.fixture
def full_tile_set() -> list[Tile]:
    return create_full_set()


@pytest.fixture
def seeded_rng() -> random.Random:
    return random.Random(42)


@pytest.fixture
def make_game():
    def _factory(
        mode: GameMode = GameMode.FFA_2P,
        seed: int = 42,
    ) -> Game:
        rng = random.Random(seed)
        count = {GameMode.FFA_2P: 2, GameMode.FFA_3P: 3, GameMode.PAIRS_4P: 4}[mode]
        players = [
            Player(seat=i, strategy=RandomStrategy(rng=random.Random(seed + i)))
            for i in range(count)
        ]
        return Game(players=players, game_mode=mode, rng=rng)

    return _factory
