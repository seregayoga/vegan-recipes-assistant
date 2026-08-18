# Evaluation

The evaluation follows the course pattern: generate realistic user queries from known records, then measure whether search returns the source recipe.

```bash
make generate-evaluation  # OpenAI call; writes 60 query/recipe-ID pairs
make evaluate             # Keyword versus hybrid retrieval
make evaluate-llm         # Concise versus detailed grounded prompts
```

`ground_truth.csv` contains three generated ingredient-availability queries for each of 20 recipes. It is synthetic data, so it is useful for a repeatable regression check, not a substitute for real-user or human-labelled evaluation.

## Current results

| Evaluation | Compared | Winner |
|---|---|---|
| Retrieval | keyword: Hit Rate@5 0.833, MRR@5 0.772; hybrid: 0.533, 0.393 | `keyword` |
| LLM judge | concise: 40% good; detailed: 40% good | `concise` (tie-break) |

The application defaults to these winners through `RETRIEVAL_MODE=keyword` and `RECOMMENDATION_PROMPT=concise`. The hybrid path still combines PostgreSQL full-text and vector search with RRF and ingredient reranking; it is retained as the evaluated alternative.

The reports are [`results/retrieval_report.md`](results/retrieval_report.md) and [`results/llm/llm_report.md`](results/llm/llm_report.md). Rerun the three commands after changing the dataset, retrieval, or prompt.
