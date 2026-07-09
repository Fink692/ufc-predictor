from __future__ import annotations

from dataclasses import dataclass, field


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


@dataclass
class EloTracker:
    base_rating: float = 1500.0
    k_factor: float = 32.0
    ratings: dict[str, float] = field(default_factory=dict)

    def rating(self, fighter_id: object) -> float:
        key = str(fighter_id)
        return self.ratings.get(key, self.base_rating)

    def probability(self, fighter_a_id: object, fighter_b_id: object) -> float:
        return expected_score(self.rating(fighter_a_id), self.rating(fighter_b_id))

    def update(self, fighter_a_id: object, fighter_b_id: object, fighter_a_won: bool) -> None:
        key_a = str(fighter_a_id)
        key_b = str(fighter_b_id)
        rating_a = self.rating(key_a)
        rating_b = self.rating(key_b)
        expected_a = expected_score(rating_a, rating_b)
        score_a = 1.0 if fighter_a_won else 0.0
        delta = self.k_factor * (score_a - expected_a)
        self.ratings[key_a] = rating_a + delta
        self.ratings[key_b] = rating_b - delta
