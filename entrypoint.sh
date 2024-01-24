#!/bin/bash

python -m spacy download en_core_web_trf

alembic upgrade head

celery -A memoir.core.tasks worker -l info -c 1 &

python run.py