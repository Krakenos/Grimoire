#!/bin/bash

if [ ! -f "config/settings.yaml" ]; then
    cp config/settings.default.yaml config/settings.yaml
fi

alembic upgrade head

mode="${MODE:-all}"
concurrency="${TASK_CONCURRENCY:-8}"

if [ $mode == "all" ]; then
  echo "Running grimoire..."
  uv run celery -A grimoire.core.tasks beat &
  uv run celery -A grimoire.core.tasks worker -l info -c $concurrency -Q summarization_queue --pool=threads &
  uv run celery -A grimoire.core.tasks worker -l info -c 1 -Q celery --pool=threads &

  uv run python run.py

elif [ $mode == "api-only" ]; then
  echo "Running grimoire in api-only mode..."
  uv run python run.py

elif [ $mode == "worker-only" ]; then
  echo "Running grimoire in worker-only mode..."
  uv run celery -A grimoire.core.tasks beat &
  uv run celery -A grimoire.core.tasks worker -l info -c 1 -Q celery --pool=threads &
  uv run celery -A grimoire.core.tasks worker -l info -c $concurrency -Q summarization_queue --pool=threads

else
  echo "UNSUPPORTED MODE"
fi
