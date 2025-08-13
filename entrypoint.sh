#!/bin/bash

if [ ! -f "config/settings.yaml" ]; then
    cp config/settings.default.yaml config/settings.yaml
fi

alembic upgrade head

mode="${MODE:-all}"
concurrency="${TASK_CONCURRENCY:-8}"

if [ $mode == "all" ]; then
  echo "Running grimoire..."
  celery -A grimoire.core.tasks beat &
  celery -A grimoire.core.tasks worker -l info -c $concurrency -Q summarization_queue --pool=threads &
  celery -A grimoire.core.tasks worker -l info -c 1 -Q celery --pool=threads &

  python run.py

elif [ $mode == "api-only"]; then
  echo "Running grimoire in api-only mode..."
  python run.py

elif [ $mode == "worker-only"]; then
  echo "Running grimoire in worker-only mode..."
  celery -A grimoire.core.tasks beat &
  celery -A grimoire.core.tasks worker -l info -c 1 -Q celery --pool=threads &
  celery -A grimoire.core.tasks worker -l info -c $concurrency -Q summarization_queue --pool=threads

else
  echo "UNSUPPORTED MODE"
fi
