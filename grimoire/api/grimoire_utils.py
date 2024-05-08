from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from grimoire.db.models import Chat, Message, User


def get_users(db_session: Session, skip: int = 0, limit: int = 100) -> Sequence[User]:
    query = select(User).offset(skip).limit(limit)
    results = db_session.scalars(query).all()
    return results


def get_user(db_session: Session, user_id: int) -> User:
    query = select(User).where(User.id == user_id)
    results = db_session.scalar(query)
    return results


def get_chats(db_session: Session, user_id: int, skip: int = 0, limit: int = 100) -> Sequence[Chat]:
    query = select(Chat).where(Chat.user_id == user_id).offset(skip).limit(limit)
    results = db_session.scalars(query).all()
    return results


def get_chat(db_session: Session, user_id: int, chat_id: int) -> Chat:
    query = select(Chat).where(Chat.user_id == user_id, Chat.id == chat_id)
    results = db_session.scalar(query)
    return results


def get_messages(db_session: Session, user_id: int, chat_id: int, skip: int = 0, limit: int = 100) -> Sequence[Message]:
    query = select(Message).where(Message.chat_id == chat_id).offset(skip).limit(limit)
    results = db_session.scalars(query).all()
    return results


def get_message(db_session: Session, user_id: int, chat_id: int, message_id: int) -> Message:
    query = select(Message).where(Message.chat_id == chat_id, Message.id == message_id)
    results = db_session.scalar(query)
    return results
