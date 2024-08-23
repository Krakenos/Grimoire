#!/bin/bash

if [ ! -f "config/settings.yaml" ]; then
    cp config/settings.default.yaml config/settings.yaml
fi

alembic upgrade head

celery -A grimoire.core.tasks beat &

concurrency="${TASK_CONCURRENCY:-8}"
celery -A grimoire.core.tasks worker -l info -c $concurrency &

python run.py