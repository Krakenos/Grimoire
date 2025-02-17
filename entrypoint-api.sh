#!/bin/bash

if [ ! -f "config/settings.yaml" ]; then
    cp config/settings.default.yaml config/settings.yaml
fi

alembic upgrade head

python run.py