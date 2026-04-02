from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.game import Action, GameState


class Strategy(ABC):
    @abstractmethod
    def choose_action(self, state: GameState) -> Action:
        ...

    def on_game_start(self) -> None:
        pass

    def on_game_end(self) -> None:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
