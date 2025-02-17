#!/bin/bash

if [ ! -f "config/settings.yaml" ]; then
    cp config/settings.default.yaml config/settings.yaml
fi

alembic upgrade head

mode="${MODE:-all}"
concurrency="${TASK_CONCURRENCY:-8}"

if [ $mode == "all" ]; then
  celery -A grimoire.core.tasks beat &
  celery -A grimoire.core.tasks worker -l info -c $concurrency -Q summarization_queue --pool=threads &
  celery -A grimoire.core.tasks worker -l info -c 1 -Q celery --pool=threads &

  python run.py

elif [ $mode == "api-only"]; then
  python run.py

elif [ $mode == "worker-only"]; then
  celery -A grimoire.core.tasks beat &
  celery -A grimoire.core.tasks worker -l info -c $concurrency -Q summarization_queue --pool=threads &
  celery -A grimoire.core.tasks worker -l info -c 1 -Q celery --pool=threads &

else
  echo "UNSUPPORTED MODE"
fi
