from __future__ import annotations

from pgvector.psycopg import Vector

from .config import get_settings
from .db import connection
from .models import ParsedQuery, RecipeCandidate
from .retrieval import reciprocal_rank_fusion, rerank


class PostgresRecipeStore:
    def __init__(self, embedder):
        self.embedder = embedder

    def _text_rows(self, query: ParsedQuery, limit: int):
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT id,title,href,ingredients,preparation,normalized_ingredients
                   FROM recipes WHERE search_document @@ websearch_to_tsquery('english', %s)
                   ORDER BY ts_rank_cd(search_document, websearch_to_tsquery('english', %s)) DESC, id LIMIT %s""",
                (query.search_text, query.search_text, limit),
            )
            return cur.fetchall()

    @staticmethod
    def _candidate(row, score: float = 0, text_rank: int = 0, vector_rank: int = 0) -> RecipeCandidate:
        return RecipeCandidate(
            recipe_id=row[0],
            title=row[1],
            href=row[2],
            ingredients=row[3],
            preparation=row[4],
            normalized_ingredients=row[5],
            rrf_score=score,
            text_rank=text_rank,
            vector_rank=vector_rank,
        )

    def search(self, query: ParsedQuery, limit: int = 20, mode: str | None = None) -> list[RecipeCandidate]:
        mode = mode or get_settings().retrieval_mode
        settings = get_settings()
        text_rows = self._text_rows(query, limit)
        if mode == "keyword":
            candidates = [self._candidate(row, text_rank=index) for index, row in enumerate(text_rows, start=1)]
            return rerank(candidates, query, settings.coverage_weight, settings.missing_weight)
        if mode != "hybrid":
            raise ValueError(f"Unknown retrieval mode: {mode}")
        vector = self.embedder.encode([query.search_text])[0]
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT id,title,href,ingredients,preparation,normalized_ingredients
                   FROM recipes ORDER BY embedding <=> %s, id LIMIT %s""",
                (Vector(vector), limit),
            )
            vector_rows = cur.fetchall()
        text_ids = [row[0] for row in text_rows]
        vector_ids = [row[0] for row in vector_rows]
        scores = reciprocal_rank_fusion([text_ids, vector_ids], settings.rrf_k)
        rows = {row[0]: row for row in text_rows + vector_rows}
        candidates = [
            self._candidate(
                row,
                score,
                text_ids.index(recipe_id) + 1 if recipe_id in text_ids else 0,
                vector_ids.index(recipe_id) + 1 if recipe_id in vector_ids else 0,
            )
            for recipe_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            for row in [rows[recipe_id]]
        ]
        return rerank(candidates, query, settings.coverage_weight, settings.missing_weight)
