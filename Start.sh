#!/bin/bash

if [ ! -f "config/settings.yaml" ]; then
    echo "Settings file doesn't exist! Creating one for you."
    cp config/settings.default.yaml config/settings.yaml
fi

if [ ! -d "venv" ]; then
    echo "Venv doesn't exist! Creating one for you."
    python3 -m venv venv
fi

docker container inspect grimoire-redis
if [ $? -eq 0 ]; then
    docker start grimoire-redis
else
    docker run -d --name grimoire-redis -p 6379:6379 redis
    docker run --name grimoire-postgres -p 5432:5432 -e POSTGRES_PASSWORD=secretpassword -e POSTGRES_USER=grimoire -e POSTGRES_DB=grimoire -d postgres
fi

source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_trf
alembic upgrade head
celery -A grimoire.core.tasks worker --concurrency=1 &
python run.py