# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Grimoire

Grimoire is a RAG (Retrieval Augmented Generation) server that builds long-term memory for AI chatbots. It processes chat messages using NLP (spaCy), extracts named entities, queues summarization tasks via Celery, stores knowledge in PostgreSQL with pgvector, and returns semantically relevant entries on each request.

## Commands

### Setup
```bash
cp config/settings.default.yaml config/settings.yaml
# Edit config/settings.yaml with your values

uv sync                   # install dependencies (add --extra cuda for GPU)
docker compose -f docker/docker-compose-dev.yaml up -d   # start Redis + Postgres
uv run python -m spacy download en_core_web_trf
uv run alembic upgrade head
```

### Running
```bash
# Celery worker (summarization tasks)
uv run celery -A grimoire.core.tasks worker --loglevel=INFO --concurrency=1 -Q summarization_queue --pool=threads

# API server (port 5005)
uv run python run.py
```

### Testing
```bash
uv run python -m unittest discover tests/
uv run python -m unittest tests.unit.test_grimoire.TestFilterSimilarEntities.test_filter_with_relations
```

### Linting / formatting
```bash
uv run ruff check .
uv run ruff format .
```

### Database migrations
```bash
uv run alembic upgrade head
uv run alembic revision --autogenerate -m "description"
```

## Architecture

### Request pipeline (`POST /grimoire/get_data`)

The main entry point is `grimoire/core/grimoire.py:process_request`. Each call:

1. Resolves or creates `User` and `Chat` records.
2. Runs spaCy NLP (`en_core_web_trf`) on new messages to extract named entities, with Redis caching.
3. Generates sentence-transformer vector embeddings (default: `Alibaba-NLP/gte-base-en-v1.5`) for messages, also Redis-cached.
4. Uses fuzzy matching (`rapidfuzz`, threshold `match_distance=80`) to deduplicate similar entity names across the conversation (e.g. "Alex" / "Alexy" → "Alex").
5. Saves new `Message`, `SpacyNamedEntity`, and `Knowledge` records; links messages to knowledge entries.
6. Enqueues Celery tasks (`describe_entity`, `generate_segmented_memory`) on `summarization_queue` for async LLM summarization.
7. Returns ranked `Knowledge`/`SegmentedMemory` entries via cosine similarity search against current message embeddings.

### Background summarization (`grimoire/core/tasks.py`)

Celery workers consume `summarization_queue`. The `describe_entity` task:
- Builds a prompt from surrounding messages + previous summary + character info.
- Calls the configured LLM backend (KoboldAI, KoboldCPP, Tabby, Aphrodite, GenericOAI).
- Saves the summary text and its vector embedding back to the `Knowledge` row.

`generate_segmented_memory` summarizes fixed message windows into `SegmentedMemory` rows.

### Key modules

| Path | Role |
|---|---|
| `grimoire/core/grimoire.py` | Core pipeline: NLP, entity dedup, save, retrieval |
| `grimoire/core/tasks.py` | Celery tasks for async LLM summarization |
| `grimoire/core/settings.py` | Pydantic settings loaded from `config/settings.yaml` |
| `grimoire/core/vector_embeddings.py` | Sentence-transformer embeddings with Redis cache |
| `grimoire/common/llm_helpers.py` | LLM API calls, tokenization (local HF or remote API) |
| `grimoire/db/models.py` | SQLAlchemy ORM: `Knowledge`, `Message`, `Chat`, `User`, `Character`, `SegmentedMemory` |
| `grimoire/db/queries.py` | Semantic search (cosine sim via pgvector + weighted recency) |
| `grimoire/api/routers/grimoire.py` | All FastAPI endpoints |

### Data model highlights

- **`Knowledge`**: one row per named entity per chat; holds `summary_entry` (formatted `[ Entity: text ]`), `token_count`, `vector_embedding`, `frozen`/`enabled` flags, and `update_count`.
- **`Message`**: stores encrypted text (or just an `external_id` when `secondary_database` is enabled), spaCy entities, and a vector embedding.
- **`SegmentedMemory`**: periodic rolling summaries across N messages, returned alongside knowledge entries.
- Sensitive columns (entity names, message text, summaries) are encrypted at rest via `SQLAlchemy-Utils` `StringEncryptedType` using `ENCRYPTION_KEY`.

### Configuration

`config/settings.yaml` (copied from `settings.default.yaml`). Supports `!env VAR_NAME` YAML tag to pull values from environment variables. Key settings:

- `summarization_api`: backend type, URL, auth, instruct format sequences
- `summarization`: prompt templates, generation params, `limit_rate` (min messages before summarizing)
- `secondary_database`: optional separate DB where raw messages are stored (for privacy); only external IDs are stored in the main DB
- `match_distance`: fuzzy match threshold (0–100) for entity deduplication
- `prefer_gpu`: controls spaCy and embedding model device

### Docker

`docker/docker-compose-dev.yaml` runs Redis on port **6370** and Postgres (pgvector) on port **5430**. Default DB credentials: `grimoire/secretpassword`.
