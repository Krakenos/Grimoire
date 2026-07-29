FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

# Keep the updates and the installs in one RUN. Split across layers, a cached `apt-get update`
# keeps serving a months-old package index while Ubuntu and deadsnakes delete the superseded
# .debs it names, so the install layer 404s on every file it asks for.
RUN apt-get update -y \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

FROM base AS deps
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra cuda
RUN uv run python -m spacy download en_core_web_trf

FROM deps AS test_deps
RUN uv sync --frozen

FROM deps AS project
COPY /grimoire ./grimoire
COPY /config ./config
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh
COPY alembic.ini ./alembic.ini
COPY run.py ./run.py

FROM test_deps AS testing
COPY /grimoire ./grimoire
COPY /tests ./tests
COPY /config ./config

FROM project AS run
EXPOSE 5005
ENTRYPOINT ["./entrypoint.sh"]
