CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS recipes (
  id text PRIMARY KEY,
  href text NOT NULL UNIQUE,
  title text NOT NULL,
  ingredients jsonb NOT NULL,
  normalized_ingredients text[] NOT NULL,
  normalized_ingredient_text text NOT NULL,
  preparation text NOT NULL,
  search_document tsvector GENERATED ALWAYS AS (
    setweight(to_tsvector('english', coalesce(title,'')), 'A') ||
    setweight(to_tsvector('english', coalesce(normalized_ingredient_text,'')), 'A') ||
    setweight(to_tsvector('english', coalesce(preparation,'')), 'C')
  ) STORED,
  embedding vector(384) NOT NULL,
  source_checksum text NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS recipes_search_idx ON recipes USING gin(search_document);
CREATE INDEX IF NOT EXISTS recipes_embedding_idx ON recipes USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS interactions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), created_at timestamptz NOT NULL DEFAULT now(),
  raw_query text NOT NULL, parsed_query jsonb, candidates jsonb, selected_recipe_ids text[],
  status text NOT NULL, error text, model text, rewrite_ms integer, retrieval_ms integer,
  llm_ms integer, total_ms integer, input_tokens integer DEFAULT 0, output_tokens integer DEFAULT 0,
  estimated_cost_usd numeric(12,8) DEFAULT 0, feedback smallint CHECK (feedback IN (-1,1)),
  feedback_at timestamptz
);
CREATE INDEX IF NOT EXISTS interactions_created_idx ON interactions(created_at);

