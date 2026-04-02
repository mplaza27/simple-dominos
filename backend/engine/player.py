from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from engine.tile import Tile

if TYPE_CHECKING:
    from strategies.base import Strategy


@dataclass
class Player:
    seat: int
    strategy: Strategy
    name: str = ""
    hand: list[Tile] = field(default_factory=list)

    @property
    def tile_count(self) -> int:
        return len(self.hand)

    @property
    def pip_sum(self) -> int:
        return sum(t.pip_sum for t in self.hand)

    @property
    def has_tiles(self) -> bool:
        return len(self.hand) > 0

    def remove_tile(self, tile: Tile) -> None:
        self.hand.remove(tile)

    def playable_tiles(self, left_end: int, right_end: int) -> list[Tile]:
        return [
            t for t in self.hand
            if t.has_value(left_end) or t.has_value(right_end)
        ]
