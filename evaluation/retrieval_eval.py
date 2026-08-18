"""Compare keyword and hybrid recipe retrieval with generated ground truth."""

import csv
import json
from pathlib import Path

from vegan_recipes.embeddings import get_embedder
from vegan_recipes.ingredients import deterministic_parse
from vegan_recipes.models import ParsedQuery
from vegan_recipes.store import PostgresRecipeStore

ROOT = Path(__file__).parent
GROUND_TRUTH = ROOT / "ground_truth.csv"
RESULTS = ROOT / "results"
MODES = ("keyword", "hybrid")


def parse_query(text: str) -> ParsedQuery:
    ingredients, exclusions = deterministic_parse(text)
    return ParsedQuery(ingredients=ingredients, exclusions=exclusions, search_text=" ".join(ingredients) or text)


def metrics(ranks: list[int | None]) -> dict[str, float]:
    return {
        "hit_rate_at_5": sum(rank is not None for rank in ranks) / len(ranks),
        "mrr_at_5": sum(1 / rank for rank in ranks if rank is not None) / len(ranks),
    }


def main() -> None:
    with GROUND_TRUTH.open(newline="") as handle:
        labels = list(csv.DictReader(handle))
    if not labels:
        raise SystemExit("Run `make generate-evaluation` first.")
    store = PostgresRecipeStore(get_embedder())
    results = {}
    for mode in MODES:
        ranks = []
        for label in labels:
            ids = [item.recipe_id for item in store.search(parse_query(label["query"]), mode=mode)[:5]]
            ranks.append(ids.index(label["recipe_id"]) + 1 if label["recipe_id"] in ids else None)
        results[mode] = metrics(ranks)
    winner = max(MODES, key=lambda mode: (results[mode]["hit_rate_at_5"], results[mode]["mrr_at_5"]))
    RESULTS.mkdir(exist_ok=True)
    summary = {"queries": len(labels), "winner": winner, "results": results}
    (RESULTS / "retrieval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Retrieval evaluation",
        "",
        f"Generated queries: {len(labels)}",
        "",
        "| Mode | Hit Rate@5 | MRR@5 |",
        "|---|---:|---:|",
    ]
    lines.extend(f"| {mode} | {values['hit_rate_at_5']:.3f} | {values['mrr_at_5']:.3f} |" for mode, values in results.items())
    lines += ["", f"Selected mode: `{winner}` (highest Hit Rate@5, then MRR@5)."]
    (RESULTS / "retrieval_report.md").write_text("\n".join(lines) + "\n")
    print(f"Selected retrieval mode: {winner}")


if __name__ == "__main__":
    main()
