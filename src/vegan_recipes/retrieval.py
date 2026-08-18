from collections.abc import Iterable

from .ingredients import PANTRY, ingredient_matches
from .models import ParsedQuery, RecipeCandidate


def reciprocal_rank_fusion(rankings: Iterable[list[str]], k: int = 30) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, recipe_id in enumerate(ranking, 1):
            scores[recipe_id] = scores.get(recipe_id, 0) + 1 / (k + rank)
    return scores


def rerank(
    candidates: list[RecipeCandidate], query: ParsedQuery, coverage_weight: float = 2.0, missing_weight: float = 0.55
) -> list[RecipeCandidate]:
    result = []
    for candidate in candidates:
        normalized = candidate.normalized_ingredients
        if any(any(ingredient_matches(x, r) for r in normalized) for x in query.exclusions):
            continue
        matched = [q for q in query.ingredients if any(ingredient_matches(q, r) for r in normalized)]
        missing = [r for r in normalized if r not in PANTRY and not any(ingredient_matches(q, r) for q in query.ingredients)]
        coverage = len(matched) / max(1, len(query.ingredients))
        candidate.matched_ingredients = matched
        candidate.missing_ingredients = missing
        candidate.final_score = candidate.rrf_score + coverage_weight * coverage - missing_weight * len(missing)
        result.append(candidate)
    return sorted(result, key=lambda c: (-c.final_score, c.title))
