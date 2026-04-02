from __future__ import annotations

from enum import Enum
from typing import Literal


class GameMode(Enum):
    FFA_2P = "ffa_2p"
    FFA_3P = "ffa_3p"
    PAIRS_4P = "pairs_4p"


End = Literal["left", "right"]


class InvalidMoveError(Exception):
    pass
