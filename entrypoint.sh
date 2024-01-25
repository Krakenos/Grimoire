#!/bin/bash

alembic upgrade head

celery -A memoir.core.tasks worker -l info -c 1 &

python run.py