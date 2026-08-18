"""Generate natural ingredient queries linked to stable recipe IDs."""

import csv
import json
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from vegan_recipes.config import get_settings
from vegan_recipes.db import connection

OUTPUT = Path(__file__).parent / "ground_truth.csv"
RECIPE_COUNT = 20


class Queries(BaseModel):
    queries: list[str] = Field(min_length=3, max_length=3)


def main() -> None:
    settings = get_settings()
    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required to generate evaluation examples.")
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, title, normalized_ingredients FROM recipes ORDER BY id LIMIT %s", (RECIPE_COUNT,))
        recipes = cur.fetchall()
    client = OpenAI(api_key=settings.openai_api_key)
    rows = []
    for recipe_id, title, ingredients in recipes:
        response = client.responses.parse(
            model=settings.openai_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Write three natural, short queries where a home cook states ingredients they have. Use different wording "
                        "from the recipe. Each query must name 2-5 ingredients. Do not ask a question, use "
                        "quantities, mention a title, or ask for instructions."
                    ),
                },
                {"role": "user", "content": json.dumps({"title": title, "ingredients": ingredients[:12]})},
            ],
            text_format=Queries,
        )
        if not response.output_parsed:
            raise RuntimeError(f"No queries generated for {recipe_id}")
        rows.extend({"query": query, "recipe_id": recipe_id, "title": title} for query in response.output_parsed.queries)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query", "recipe_id", "title"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} generated queries to {OUTPUT}")


if __name__ == "__main__":
    main()
