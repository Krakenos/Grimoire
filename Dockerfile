ARG PYTHON_VERSION=3.10.12
FROM python:${PYTHON_VERSION}-slim as base
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

FROM base as deps
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN python -m spacy download en_core_web_trf

FROM deps as project
COPY . .
EXPOSE 5005
ENTRYPOINT ["./entrypoint.sh"]
