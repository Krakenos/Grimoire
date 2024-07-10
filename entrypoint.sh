#!/bin/bash

alembic upgrade head

concurrency="${TASK_CONCURRENCY:-8}"
celery -A grimoire.core.tasks worker -l info -c $concurrency &

python run.py