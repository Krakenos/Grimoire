ARG PYTHON_VERSION=3.10.12

FROM python:${PYTHON_VERSION}-slim as base
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

FROM base AS deps
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN python -m spacy download en_core_web_trf

FROM deps AS test_deps
COPY requirements_tests.txt requirements_tests.txt
RUN pip install -r requirements_tests.txt

FROM deps AS project
COPY /grimoire ./grimoire
COPY /config ./config
COPY entrypoint.sh ./entrypoint.sh
COPY alembic.ini ./alembic.ini
COPY run.py ./run.py

FROM test_deps AS testing
COPY /grimoire ./grimoire
COPY /tests ./tests
COPY /config ./config

FROM project as run
EXPOSE 5005
ENTRYPOINT ["./entrypoint.sh"]
