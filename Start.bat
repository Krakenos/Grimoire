@echo off

cd "%~dp0"

CALL python -V
IF ERRORLEVEL 1 (
    ECHO Can't find python command, make sure python is installed and added to PATH environmental variable
    PAUSE
    EXIT
)

IF NOT EXIST "venv\" (
    ECHO Creating venv...
    CALL python -m venv venv
)

CALL docker container inspect grimoire-redis
IF ERRORLEVEL 1 (
    CALL docker run -d --name grimoire-redis -p 6379:6379 redis
    CALL docker run --name grimoire-postgres -p 5432:5432 -e POSTGRES_PASSWORD=secretpassword -e POSTGRES_USER=grimoire -e POSTGRES_DB=grimoire -d postgres
    IF ERRORLEVEL 1 (
        ECHO Error, docker is either not installed or not running.
        PAUSE
        EXIT
    )
) ELSE (
    CALL docker start grimoire-redis
)
CALL .\venv\Scripts\activate.bat
CALL pip install -r requirements.txt
CALL pip install eventlet

IF NOT EXIST "venv\Lib\site-packages\en_core_web_trf" (
    CALL python -m spacy download en_core_web_trf
)

CALL alembic upgrade head
START celery -A grimoire.core.tasks worker --loglevel=INFO --concurrency=1 -P eventlet
CALL python run.py