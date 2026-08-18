"""Compare the two deployable recommendation prompts with an LLM judge."""

import csv
import json
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from vegan_recipes.assistant import parse_query, recommend_with_usage
from vegan_recipes.config import get_settings
from vegan_recipes.embeddings import get_embedder
from vegan_recipes.store import PostgresRecipeStore

ROOT = Path(__file__).parent
GROUND_TRUTH = ROOT / "ground_truth.csv"
OUTPUT = ROOT / "results" / "llm"
PROMPT_VERSIONS = ("concise", "detailed")
HOLDOUT_SIZE = 10


class Judgement(BaseModel):
    score: Literal["good", "bad"]
    reason: str = Field(max_length=300)


def judge(client: OpenAI, query: str, title: str, answer: dict) -> Judgement:
    response = client.responses.parse(
        model=get_settings().openai_model,
        input=[
            {
                "role": "system",
                "content": (
                    "Judge whether the recipe answer is useful and grounded in its supplied recipes. "
                    "Return good only when it recommends existing recipes and directly addresses the ingredient query."
                ),
            },
            {"role": "user", "content": json.dumps({"query": query, "expected_recipe": title, "answer": answer})},
        ],
        text_format=Judgement,
    )
    if not response.output_parsed:
        raise RuntimeError("Judge returned no result")
    return response.output_parsed


def main() -> None:
    settings = get_settings()
    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required for LLM evaluation.")
    with GROUND_TRUTH.open(newline="") as handle:
        cases = list(csv.DictReader(handle))[-HOLDOUT_SIZE:]
    if len(cases) < HOLDOUT_SIZE:
        raise SystemExit(f"Need at least {HOLDOUT_SIZE} generated queries.")
    client = OpenAI(api_key=settings.openai_api_key)
    store = PostgresRecipeStore(get_embedder())
    records, scores = [], {version: [] for version in PROMPT_VERSIONS}
    for case in cases:
        parsed = parse_query(case["query"])
        candidates = store.search(parsed)
        for version in PROMPT_VERSIONS:
            answer, usage = recommend_with_usage(parsed, candidates, client, prompt=version)
            if usage.error:
                scores[version].append(False)
                records.append({"query": case["query"], "prompt": version, "score": "bad", "reason": usage.error})
                continue
            result = judge(client, case["query"], case["title"], answer.model_dump())
            scores[version].append(result.score == "good")
            records.append({"query": case["query"], "prompt": version, "score": result.score, "reason": result.reason})
    rates = {version: sum(values) / len(values) for version, values in scores.items()}
    winner = max(PROMPT_VERSIONS, key=lambda version: (rates[version], version == "concise"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    lines = ["# LLM evaluation", "", f"Held-out generated queries: {len(cases)}", "", "| Prompt | Good answers |", "|---|---:|"]
    lines.extend(f"| {version} | {rate:.0%} |" for version, rate in rates.items())
    lines += ["", f"Selected prompt: `{winner}` (highest good-answer rate; ties select concise)."]
    (OUTPUT / "llm_report.md").write_text("\n".join(lines) + "\n")
    print(f"Selected recommendation prompt: {winner}")


if __name__ == "__main__":
    main()
