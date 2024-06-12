#!/bin/bash

alembic upgrade head

celery -A grimoire.core.tasks worker -l info &

python run.py