from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from grimoire.core.settings import settings

engine = create_engine(settings["DB_ENGINE"])

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
