from vegan_recipes.assistant import (
    ModelUsage,
    deterministic_response,
    estimate_cost,
    parse_query,
    parse_query_with_usage,
    recommend,
    recommend_with_usage,
)
from vegan_recipes.ingredients import normalize_ingredient, split_ingredients
from vegan_recipes.models import ParsedQuery, RecipeCandidate
from vegan_recipes.retrieval import reciprocal_rank_fusion, rerank


def candidate(recipe_id="a", normalized=None):
    return RecipeCandidate(
        recipe_id=recipe_id,
        title="Dinner",
        href="https://example.com/a",
        ingredients=["1 cup chickpeas", "salt"],
        normalized_ingredients=normalized or ["chickpea", "salt"],
        preparation="Cook everything.",
    )


def test_normalization_alias_quantity_and_split():
    assert normalize_ingredient("2 cups garbanzo beans") == "chickpea"
    assert split_ingredients("tomato; spinach\nchickpeas") == ["tomato", "spinach", "chickpeas"]


def test_parse_fallback_and_exclusion():
    parsed = parse_query("I have chickpeas, tomato, no peanuts")
    assert "chickpea" in parsed.ingredients
    assert "peanut" in parsed.exclusions or "peanuts" in parsed.exclusions


def test_rrf():
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=10)
    assert scores["a"] == scores["b"]


def test_rerank_filters_exclusions_and_ignores_pantry_missing():
    query = ParsedQuery(ingredients=["chickpea"], exclusions=["peanut"], search_text="chickpea")
    ranked = rerank([candidate("good"), candidate("bad", ["chickpea", "peanut"])], query)
    assert [x.recipe_id for x in ranked] == ["good"]
    assert ranked[0].missing_ingredients == []


def test_deterministic_response_uses_existing_ids():
    item = candidate()
    item.matched_ingredients = ["chickpea"]
    answer = deterministic_response([item])
    assert answer.recommendations[0].recipe_id == "a"


def test_recommend_returns_no_match_when_candidates_have_no_ingredient_overlap():
    query = ParsedQuery(ingredients=["mango"], search_text="mango")
    answer = recommend(query, [candidate()], client=None)
    assert answer.status == "no_match"
    assert answer.recommendations == []


def test_deterministic_model_usage_and_recommendation_metadata():
    parsed, parse_usage = parse_query_with_usage("chickpeas, tomato")
    item = candidate()
    item.matched_ingredients = ["chickpea"]
    answer, recommendation_usage = recommend_with_usage(parsed, [item], client=None)
    assert answer.status == "ok"
    assert parse_usage == ModelUsage()
    assert recommendation_usage == ModelUsage()


def test_cost_calculation(monkeypatch):
    from vegan_recipes.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("INPUT_COST_PER_MILLION", "1")
    monkeypatch.setenv("OUTPUT_COST_PER_MILLION", "2")
    assert estimate_cost(1_000_000, 500_000) == 2
    get_settings.cache_clear()
