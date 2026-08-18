import ast
import hashlib
import io
import os
import zipfile

import httpx
import pandas as pd
from pgvector.psycopg import Vector
from prefect import flow, task
from psycopg.types.json import Jsonb

from vegan_recipes.db import connection, migrate
from vegan_recipes.embeddings import get_embedder
from vegan_recipes.ingredients import normalize_ingredient, split_ingredients

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/rodrigoazs/vegan-recipes"
# Override only for a deliberately reviewed dataset update.
EXPECTED_SHA256 = os.getenv("DATASET_SHA256", "44d16f9ef1a59bb2156f88ea711566312b59ad64c1a3cb9e36914ef2cc318ca8")
EXPECTED_ROWS = 1390


def stable_id(href: str) -> str:
    return hashlib.sha256(href.strip().encode()).hexdigest()[:24]


@task(retries=3, retry_delay_seconds=5)
def download() -> bytes:
    response = httpx.get(DATASET_URL, follow_redirects=True, timeout=120)
    response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest()
    if EXPECTED_SHA256 and digest != EXPECTED_SHA256:
        raise ValueError(f"dataset checksum mismatch: {digest}")
    return response.content


@task
def transform(payload: bytes) -> tuple[list[dict], str]:
    digest = hashlib.sha256(payload).hexdigest()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_names = [n for n in archive.namelist() if n.endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"expected one CSV, found {csv_names}")
        frame = pd.read_csv(archive.open(csv_names[0]))
    frame = frame.drop(columns=[c for c in frame.columns if c.lower().startswith("unnamed")])
    required = {"href", "title", "ingredients", "preparation"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing columns: {required - set(frame.columns)}")
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, got {len(frame)}")
    records = []
    for row in frame.to_dict("records"):
        raw = row["ingredients"]
        try:
            parsed = ast.literal_eval(raw) if isinstance(raw, str) and raw.startswith("[") else raw
        except (ValueError, SyntaxError):
            parsed = raw
        ingredients = split_ingredients(parsed)
        normalized = list(
            dict.fromkeys(filter(None, (normalize_ingredient(x) for x in ingredients)))
        )
        records.append(
            {
                "id": stable_id(row["href"]),
                "href": row["href"].strip(),
                "title": " ".join(str(row["title"]).split()),
                "ingredients": ingredients,
                "normalized": normalized,
                "preparation": "\n".join(
                    line.strip() for line in str(row["preparation"]).splitlines() if line.strip()
                ),
            }
        )
    return records, digest


@task
def load(records: list[dict], checksum: str) -> int:
    migrate()
    vectors = get_embedder().encode(
        [r["title"] + ". " + ", ".join(r["normalized"]) for r in records]
    )
    with connection() as conn, conn.cursor() as cur:
        for record, vector in zip(records, vectors, strict=True):
            cur.execute(
                """INSERT INTO recipes
              (id,href,title,ingredients,normalized_ingredients,normalized_ingredient_text,preparation,embedding,source_checksum)
              VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO UPDATE SET
              href=excluded.href,title=excluded.title,ingredients=excluded.ingredients,
              normalized_ingredients=excluded.normalized_ingredients,preparation=excluded.preparation,
              normalized_ingredient_text=excluded.normalized_ingredient_text,
              embedding=excluded.embedding,source_checksum=excluded.source_checksum,updated_at=now()""",
                (
                    record["id"],
                    record["href"],
                    record["title"],
                    Jsonb(record["ingredients"]),
                    record["normalized"],
                    " ".join(record["normalized"]),
                    record["preparation"],
                    Vector(vector),
                    checksum,
                ),
            )
        cur.execute("SELECT count(*) FROM recipes")
        return cur.fetchone()[0]


@flow(name="vegan-recipes-ingestion", log_prints=True)
def ingest() -> int:
    payload = download()
    records, checksum = transform(payload)
    count = load(records, checksum)
    print(f"Loaded {count} recipes; source SHA256={checksum}")
    return count


if __name__ == "__main__":
    ingest()
