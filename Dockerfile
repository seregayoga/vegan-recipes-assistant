FROM python:3.12.11-slim@sha256:47ae396f09c1303b8653019811a8498470603d7ffefc29cb07c88f1f8cb3d19f
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir uv==0.8.11
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev
COPY . .
ENV PYTHONPATH=/app/src PATH=/app/.venv/bin:$PATH
RUN /app/.venv/bin/python -c "import httpx, pandas, prefect, psycopg, streamlit"
EXPOSE 8501
CMD ["/app/.venv/bin/streamlit", "run", "app.py", "--server.address=0.0.0.0"]

