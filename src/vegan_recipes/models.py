from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


class ParsedQuery(BaseModel):
    ingredients: list[str] = Field(default_factory=list, max_length=50)
    exclusions: list[str] = Field(default_factory=list, max_length=20)
    preferences: list[str] = Field(default_factory=list, max_length=20)
    search_text: str = Field(min_length=1, max_length=500)

    @field_validator("ingredients", "exclusions", "preferences")
    @classmethod
    def unique_normalized(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(v.strip().lower() for v in values if v.strip()))


class RecipeCandidate(BaseModel):
    recipe_id: str
    title: str
    href: HttpUrl
    ingredients: list[str]
    preparation: str
    normalized_ingredients: list[str]
    matched_ingredients: list[str] = Field(default_factory=list)
    missing_ingredients: list[str] = Field(default_factory=list)
    text_rank: float = 0
    vector_rank: float = 0
    rrf_score: float = 0
    final_score: float = 0


class Recommendation(BaseModel):
    recipe_id: str
    explanation: str = Field(max_length=400)
    matched_ingredients: list[str]
    missing_ingredients: list[str]
    preparation_summary: str = Field(max_length=600)


class AssistantResponse(BaseModel):
    status: Literal["ok", "no_match", "invalid"]
    message: str = Field(max_length=500)
    recommendations: list[Recommendation] = Field(default_factory=list, max_length=3)
