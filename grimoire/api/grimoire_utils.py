from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from grimoire.db.models import Chat, Knowledge, Message, User


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
    query = (
        select(Message)
        .join(Message.chat)
        .where(Message.chat_id == chat_id, Chat.user_id == user_id)
        .offset(skip)
        .limit(limit)
    )
    results = db_session.scalars(query).all()
    return results


def get_message(db_session: Session, user_id: int, chat_id: int, message_index: int) -> Message:
    query = (
        select(Message)
        .join(Message.chat)
        .where(Message.chat_id == chat_id, Message.message_index == message_index, Chat.user_id == user_id)
    )
    results = db_session.scalar(query)
    return results


def get_all_knowledge(
    db_session: Session, user_id: int, chat_id: int, skip: int = 0, limit: int = 100
) -> Sequence[Knowledge]:
    query = (
        select(Knowledge)
        .join(Knowledge.chat)
        .where(Knowledge.chat_id == chat_id, Chat.user_id == user_id, Knowledge.summary.is_not(None))
        .offset(skip)
        .limit(limit)
    )
    results = db_session.scalars(query).all()
    return results


def get_knowledge(db_session: Session, user_id: int, chat_id: int, knowledge_id: int) -> Knowledge:
    query = (
        select(Knowledge)
        .join(Knowledge.chat)
        .where(
            Knowledge.chat_id == chat_id,
            Knowledge.id == knowledge_id,
            Chat.user_id == user_id,
            Knowledge.summary.is_not(None),
        )
    )
    results = db_session.scalar(query)
    return results


def get_user_by_external(db_session: Session, external_id: str) -> User | None:
    query = select(User).where(User.external_id == external_id)
    result = db_session.scalar(query)
    return result


def get_chat_by_external(db_session: Session, external_id: str, user_id: int) -> Chat | None:
    query = select(Chat).where(Chat.external_id == external_id, Chat.user_id == user_id)
    result = db_session.scalar(query)
    return result
