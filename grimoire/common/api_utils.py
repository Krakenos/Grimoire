from collections.abc import Sequence
from datetime import datetime
from typing import TypeVar

from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from grimoire.common.llm_helpers import token_count
from grimoire.core.settings import settings
from grimoire.core.vector_embeddings import get_text_embeddings
from grimoire.db.models import Base, Chat, Knowledge, Message, User

ORMBase = TypeVar("ORMBase", bound=Base)


def get_users(db_session: Session, skip: int = 0, limit: int = 100) -> Sequence[User]:
    query = select(User).offset(skip).limit(limit)
    results = db_session.scalars(query).all()
    return results


def get_user(db_session: Session, user_id: int) -> User | None:
    query = select(User).where(User.id == user_id)
    results = db_session.scalar(query)
    return results


def create_user(db: Session, external_id: str) -> User:
    new_user = User(external_id=external_id)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


def get_chats(db_session: Session, user_id: int, skip: int = 0, limit: int = 100) -> Sequence[Chat]:
    query = select(Chat).where(Chat.user_id == user_id).offset(skip).limit(limit)
    results = db_session.scalars(query).all()
    return results


def get_chat(db_session: Session, user_id: int, chat_id: int) -> Chat | None:
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


def get_message(db_session: Session, user_id: int, chat_id: int, message_index: int) -> Message | None:
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


def get_knowledge(db_session: Session, user_id: int, chat_id: int, knowledge_id: int) -> Knowledge | None:
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


# TODO change to inline TypeVar after updating to newer python
def update_record(db: Session, db_object: ORMBase, request_object: BaseModel) -> ORMBase:
    new_attributes = request_object.model_dump(exclude_unset=True, exclude_none=True, exclude_defaults=True)
    for key, value in new_attributes.items():
        setattr(db_object, key, value)
    db.add(db_object)
    db.commit()
    return db_object


# TODO this is manual cleanup as a quick fix, check if this can be done properly with cascades
def delete_chat(db_session: Session, chat: Chat) -> None:
    for message in chat.messages:
        for named_entity in message.spacy_named_entities:
            db_session.delete(named_entity)
        db_session.delete(message)
    db_session.commit()
    db_session.refresh(chat)

    for character in chat.characters:
        for trigger_text in character.trigger_texts:
            db_session.delete(trigger_text)
        db_session.delete(character)
    db_session.commit()
    db_session.refresh(chat)

    stmt = delete(Knowledge).where(Knowledge.chat_id == chat.id)
    db_session.execute(stmt)
    db_session.refresh(chat)
    db_session.delete(chat)
    db_session.commit()


def delete_user(db_session: Session, user: User) -> None:
    for chat in user.chats:
        delete_chat(db_session, chat)
    db_session.refresh(user)
    db_session.delete(user)
    db_session.commit()


def update_summary_metadata(db_session: Session, knowledge: Knowledge) -> Knowledge:
    """
    Updates summary_entry, token count, update_date and vector embedding for knowledge
    """
    knowledge.summary_entry = f"[ {knowledge.entity}: {knowledge.summary} ]"
    knowledge.token_count = token_count(
        [knowledge.summary_entry],
        settings.summarization_api.backend,
        settings.summarization_api.url,
        settings.tokenization.local_tokenizer,
        settings.tokenization.prefer_local_tokenizer,
        settings.summarization_api.auth_key,
    )[0]
    knowledge.updated_date = datetime.now()
    knowledge.vector_embedding = get_text_embeddings(knowledge.summary)[0]
    db_session.add(knowledge)
    db_session.commit()
    db_session.refresh(knowledge)
    return knowledge
