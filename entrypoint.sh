#!/bin/bash

alembic upgrade head

celery -A grimoire.core.tasks worker -l info -c 1 &

python run.py