from __future__ import annotations

from dataclasses import dataclass, field

from engine.types import End, InvalidMoveError


@dataclass(frozen=True, slots=True)
class Tile:
    high: int
    low: int

    def __init__(self, a: int, b: int) -> None:
        hi, lo = (a, b) if a >= b else (b, a)
        object.__setattr__(self, "high", hi)
        object.__setattr__(self, "low", lo)

    @property
    def pip_sum(self) -> int:
        return self.high + self.low

    @property
    def is_double(self) -> bool:
        return self.high == self.low

    def has_value(self, value: int) -> bool:
        return self.high == value or self.low == value

    def other_value(self, value: int) -> int:
        if self.high == value:
            return self.low
        if self.low == value:
            return self.high
        raise ValueError(f"Tile {self} does not have value {value}")

    def __repr__(self) -> str:
        return f"[{self.low}|{self.high}]"


@dataclass
class Board:
    _left_end: int | None = field(default=None, init=False)
    _right_end: int | None = field(default=None, init=False)
    _tiles: list[Tile] = field(default_factory=list, init=False)

    @property
    def is_empty(self) -> bool:
        return self._left_end is None

    @property
    def ends(self) -> tuple[int, int] | None:
        if self.is_empty:
            return None
        assert self._left_end is not None and self._right_end is not None
        return (self._left_end, self._right_end)

    @property
    def played_tiles(self) -> frozenset[Tile]:
        return frozenset(self._tiles)

    def can_play(self, tile: Tile) -> bool:
        if self.is_empty:
            return True
        assert self._left_end is not None and self._right_end is not None
        return tile.has_value(self._left_end) or tile.has_value(self._right_end)

    def valid_placements(self, tile: Tile) -> list[End]:
        if self.is_empty:
            return ["left"]
        assert self._left_end is not None and self._right_end is not None
        placements: list[End] = []
        if tile.has_value(self._left_end):
            placements.append("left")
        if tile.has_value(self._right_end):
            if self._left_end != self._right_end or not placements:
                placements.append("right")
            elif tile.is_double:
                placements.append("right")
        return placements

    def play(self, tile: Tile, end: End) -> None:
        if self.is_empty:
            self._left_end = tile.low
            self._right_end = tile.high
            self._tiles.append(tile)
            return

        assert self._left_end is not None and self._right_end is not None

        if end == "left":
            if not tile.has_value(self._left_end):
                raise InvalidMoveError(
                    f"Tile {tile} cannot be played on left end {self._left_end}"
                )
            self._left_end = tile.other_value(self._left_end)
            self._tiles.append(tile)
        elif end == "right":
            if not tile.has_value(self._right_end):
                raise InvalidMoveError(
                    f"Tile {tile} cannot be played on right end {self._right_end}"
                )
            self._right_end = tile.other_value(self._right_end)
            self._tiles.append(tile)
        else:
            raise InvalidMoveError(f"Invalid end: {end}")


def create_full_set() -> list[Tile]:
    tiles: list[Tile] = []
    for i in range(10):
        for j in range(i, 10):
            tiles.append(Tile(i, j))
    return tiles
