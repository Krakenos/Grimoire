from sqlalchemy import select
from sqlalchemy.orm import Session

from grimoire.db.models import User


def get_users(db_session: Session, skip: int = 0, limit: int = 100):
    query = select(User).offset(skip).limit(limit)
    results = db_session.scalars(query).all()
    return results


def get_user(db_session: Session, user_id: int) -> User:
    query = select(User).where(User.id == user_id)
    results = db_session.scalar(query)
    return results
