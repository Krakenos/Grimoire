#!/bin/bash

if [ ! -d "venv" ]; then
    echo "Venv doesn't exist! Creating one for you."
    python3 -m venv venv
fi

docker container inspect grimoire-redis
if [ $? -eq 0 ]; then
    docker start grimoire-redis
else
    docker run -d --name grimoire-redis -p 6379:6379 redis
fi

source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_trf
alembic upgrade head
celery -A grimoire.core.tasks worker --concurrency=1 &
python run.py