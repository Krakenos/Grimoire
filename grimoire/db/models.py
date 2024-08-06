from datetime import datetime

from sqlalchemy import Column, ForeignKey, Table, Unicode
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy_utils import StringEncryptedType
from sqlalchemy_utils.types.encrypted.encrypted_type import AesEngine

from grimoire.core.settings import settings

encryption_key = settings["ENCRYPTION_KEY"]


class Base(DeclarativeBase):
    pass


knowledge_message = Table(
    "knowledge_message",
    Base.metadata,
    Column("knowledge_id", ForeignKey("knowledge.id"), primary_key=True),
    Column("message_id", ForeignKey("message.id"), primary_key=True),
)


class Knowledge(Base):
    __tablename__ = "knowledge"

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey("chat.id"))
    chat: Mapped["Chat"] = relationship(back_populates="knowledge")
    entity = Column(StringEncryptedType(Unicode, encryption_key, AesEngine, "pkcs5"), nullable=False)
    entity_type: Mapped[str | None]
    entity_label: Mapped[str | None]
    summary = Column(StringEncryptedType(Unicode, encryption_key, AesEngine, "pkcs5"))
    token_count: Mapped[int | None]
    messages: Mapped[list["Message"]] = relationship(secondary=knowledge_message, back_populates="knowledge")
    updated_date: Mapped[datetime] = mapped_column(default=datetime.now)
    update_at: Mapped[int | None] = mapped_column(default=1)
    update_count: Mapped[int | None] = mapped_column(default=1)

    def __repr__(self):
        return f"id: {self.id}, name: {self.entity}"


class Message(Base):
    __tablename__ = "message"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[int | None]
    chat_id: Mapped[int] = mapped_column(ForeignKey("chat.id"))
    chat: Mapped["Chat"] = relationship(back_populates="messages")
    message_index: Mapped[int]
    message = Column(StringEncryptedType(Unicode, encryption_key, AesEngine, "pkcs5"), nullable=False)
    created_date: Mapped[datetime] = mapped_column(default=datetime.now)
    spacy_named_entities: Mapped[list["SpacyNamedEntity"]] = relationship()
    knowledge: Mapped[list["Knowledge"]] = relationship(secondary=knowledge_message, back_populates="messages")


class SpacyNamedEntity(Base):
    __tablename__ = "spacy_named_entities"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("message.id"))
    entity_name = Column(StringEncryptedType(Unicode, encryption_key, AesEngine, "pkcs5"), nullable=False)
    entity_label: Mapped[str]


class Chat(Base):
    __tablename__ = "chat"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str]
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"))
    messages: Mapped[list["Message"]] = relationship()
    knowledge: Mapped[list["Knowledge"]] = relationship()


class User(Base):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str]
    chats: Mapped[list["Chat"]] = relationship()
