from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from .config import get_settings


@contextmanager
def connection():
    with psycopg.connect(get_settings().database_url) as conn:
        register_vector(conn)
        yield conn


def migrate() -> None:
    sql = (Path(__file__).parents[2] / "sql" / "001_init.sql").read_text()
    with connection() as conn, conn.cursor() as cur:
        cur.execute(sql)


def log_interaction(**values: Any) -> str:
    columns = ",".join(values)
    placeholders = ",".join(["%s"] * len(values))
    prepared = [Jsonb(v) if key in {"parsed_query", "candidates"} else v for key, v in values.items()]
    with connection() as conn, conn.cursor() as cur:
        cur.execute(f"INSERT INTO interactions ({columns}) VALUES ({placeholders}) RETURNING id", prepared)
        return str(cur.fetchone()[0])


def save_feedback(interaction_id: str, value: int) -> None:
    if value not in (-1, 1):
        raise ValueError("feedback must be -1 or 1")
    with connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE interactions SET feedback=%s, feedback_at=now() WHERE id=%s", (value, interaction_id))
