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
# Dependencies only. Grimoire's own version is read from grimoire/__init__.py
# ([tool.hatch.version]), which isn't copied yet, so building the project here would fail.
# It gets installed in the stages below, once the source is in place.
# en_core_web_trf is a pinned dependency in pyproject.toml/uv.lock, so it comes in with this
# sync - no separate `spacy download` needed.
RUN uv sync --frozen --no-dev --no-install-project --extra cuda

FROM deps AS test_deps
RUN uv sync --frozen --no-install-project

FROM deps AS project
COPY /grimoire ./grimoire
COPY /config ./config
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh
COPY alembic.ini ./alembic.ini
COPY run.py ./run.py
# pyproject.toml points readme/license.file at these; hatchling errors out if they are missing.
COPY README.md LICENSE ./
# Installs grimoire itself; the dependency layer above is reused untouched. Flags must match
# that sync, since uv makes the environment match exactly and would otherwise drop the extras.
RUN uv sync --frozen --no-dev --extra cuda

FROM test_deps AS testing
COPY /grimoire ./grimoire
COPY /tests ./tests
COPY /config ./config
COPY README.md LICENSE ./
RUN uv sync --frozen
ENV PATH="/app/.venv/bin:$PATH"

FROM project AS run
# entrypoint.sh calls `alembic upgrade head` directly, and alembic lives in the project venv.
ENV PATH="/app/.venv/bin:$PATH"
# The environment is already built above, exactly as the lockfile specifies. Without this,
# every `uv run` in entrypoint.sh re-syncs on container start: it drops the cuda extra and
# pulls the dev group, and needs network to do it.
ENV UV_NO_SYNC=1
EXPOSE 5005
ENTRYPOINT ["./entrypoint.sh"]
