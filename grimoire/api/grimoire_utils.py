from sqlalchemy import select
from sqlalchemy.orm import Session

from grimoire.db.models import User


def get_user(db_session: Session, user_id: int) -> User:
    query = select(User).where(User.id == user_id)
    results = db_session.scalar(query)
    return results
