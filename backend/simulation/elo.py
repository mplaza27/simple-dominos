from __future__ import annotations


class EloSystem:
    """Elo rating tracker with K-factor decay."""

    def __init__(self, default_rating: float = 1500.0, decay_games: int = 100) -> None:
        self._ratings: dict[str, float] = {}
        self._games_played: dict[str, int] = {}
        self._default_rating = default_rating
        self._decay_games = decay_games

    def _k_factor(self, name: str) -> float:
        played = self._games_played.get(name, 0)
        return max(16.0, 32.0 - played * 16.0 / self._decay_games)

    def _expected(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def get_rating(self, name: str) -> float:
        return self._ratings.get(name, self._default_rating)

    def get_all_ratings(self) -> dict[str, float]:
        return dict(self._ratings)

    def _ensure(self, name: str) -> None:
        if name not in self._ratings:
            self._ratings[name] = self._default_rating
            self._games_played[name] = 0

    def update(self, winner: str, loser: str, draw: bool = False) -> None:
        self._ensure(winner)
        self._ensure(loser)

        ra = self._ratings[winner]
        rb = self._ratings[loser]

        ea = self._expected(ra, rb)
        eb = self._expected(rb, ra)

        ka = self._k_factor(winner)
        kb = self._k_factor(loser)

        if draw:
            self._ratings[winner] += ka * (0.5 - ea)
            self._ratings[loser] += kb * (0.5 - eb)
        else:
            self._ratings[winner] += ka * (1.0 - ea)
            self._ratings[loser] += kb * (0.0 - eb)

        self._games_played[winner] += 1
        self._games_played[loser] += 1

    def update_multiplayer(
        self, winners: list[str], losers: list[str]
    ) -> None:
        """Pairwise Elo updates for multi-player games.

        Winners draw against each other, losers draw against each other,
        each winner beats each loser.
        """
        # Winners beat losers
        for w in winners:
            for l in losers:
                self.update(w, l, draw=False)

        # Winners draw among themselves
        for i, w1 in enumerate(winners):
            for w2 in winners[i + 1 :]:
                self.update(w1, w2, draw=True)

        # Losers draw among themselves
        for i, l1 in enumerate(losers):
            for l2 in losers[i + 1 :]:
                self.update(l1, l2, draw=True)
