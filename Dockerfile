FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime AS base
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

FROM base AS deps
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip --no-cache && pip install -r requirements.txt --no-cache
RUN python3 -m spacy download en_core_web_trf

FROM deps AS test_deps
COPY requirements_tests.txt requirements_tests.txt
RUN pip install -r requirements_tests.txt

FROM deps AS project
COPY /grimoire ./grimoire
COPY /config ./config

FROM test_deps AS testing
COPY /grimoire ./grimoire
COPY /tests ./tests
COPY /config ./config

FROM project as run
EXPOSE 5005
ENTRYPOINT ["./entrypoint.sh"]
