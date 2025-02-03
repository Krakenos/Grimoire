from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, ForeignKey, Table, Unicode
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy_utils import StringEncryptedType
from sqlalchemy_utils.types.encrypted.encrypted_type import AesEngine

from grimoire.core.settings import settings

encryption_key = settings.ENCRYPTION_KEY


class Base(DeclarativeBase):
    pass


knowledge_message = Table(
    "knowledge_message",
    Base.metadata,
    Column("knowledge_id", ForeignKey("knowledge.id"), primary_key=True),
    Column("message_id", ForeignKey("message.id"), primary_key=True),
)

segmented_memories_message = Table(
    "segmented_memories_message",
    Base.metadata,
    Column("segmented_memory_id", ForeignKey("segmented_memories.id"), primary_key=True),
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
    summary = Column(StringEncryptedType(Unicode, encryption_key, AesEngine, "pkcs5"), nullable=True)
    summary_entry = Column(StringEncryptedType(Unicode, encryption_key, AesEngine, "pkcs5"), nullable=True)
    token_count: Mapped[int | None]
    enabled: Mapped[bool] = mapped_column(default=True)
    frozen: Mapped[bool] = mapped_column(default=False)
    messages: Mapped[list["Message"]] = relationship(secondary=knowledge_message, back_populates="knowledge")
    updated_date: Mapped[datetime] = mapped_column(default=datetime.now)
    update_at: Mapped[int | None] = mapped_column(default=1)
    update_count: Mapped[int | None] = mapped_column(default=1)
    vector_embedding = mapped_column(Vector())

    def __repr__(self):
        return f"id: {self.id}, name: {self.entity}"


class Message(Base):
    __tablename__ = "message"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str | None]
    chat_id: Mapped[int] = mapped_column(ForeignKey("chat.id"))
    chat: Mapped["Chat"] = relationship(back_populates="messages")
    message_index: Mapped[int]
    character_id: Mapped[int] = mapped_column(ForeignKey("character.id"))
    character: Mapped["Character"] = relationship()
    message = Column(StringEncryptedType(Unicode, encryption_key, AesEngine, "pkcs5"), nullable=True)
    created_date: Mapped[datetime] = mapped_column(default=datetime.now)
    spacy_named_entities: Mapped[list["SpacyNamedEntity"]] = relationship()
    knowledge: Mapped[list["Knowledge"]] = relationship(secondary=knowledge_message, back_populates="messages")
    segmented_memories: Mapped[list["SegmentedMemory"]] = relationship(
        secondary=segmented_memories_message, back_populates="messages"
    )
    vector_embedding = mapped_column(Vector())


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
    characters: Mapped[list["Character"]] = relationship()
    segmented_memmories: Mapped[list["SegmentedMemory"]] = relationship()
    segmented_memory_interval: Mapped[int] = mapped_column(default=5)
    segmented_memory_messages: Mapped[int] = mapped_column(default=10)


class User(Base):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str]
    chats: Mapped[list["Chat"]] = relationship()


class Character(Base):
    __tablename__ = "character"

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey("chat.id"))
    name = Column(StringEncryptedType(Unicode, encryption_key, AesEngine, "pkcs5"), nullable=False)
    description = Column(StringEncryptedType(Unicode, encryption_key, AesEngine, "pkcs5"), nullable=True)
    character_note = Column(StringEncryptedType(Unicode, encryption_key, AesEngine, "pkcs5"), nullable=True)
    trigger_texts: Mapped[list["CharacterTriggerText"]] = relationship()


class CharacterTriggerText(Base):
    __tablename__ = "character_trigger_text"

    id: Mapped[int] = mapped_column(primary_key=True)
    character_id: Mapped[int] = mapped_column(ForeignKey("character.id"))
    text = Column(StringEncryptedType(Unicode, encryption_key, AesEngine, "pkcs5"), nullable=False)


class SegmentedMemory(Base):
    __tablename__ = "segmented_memories"

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey("chat.id"))
    chat: Mapped[Chat] = relationship(back_populates="segmented_memmories")
    messages: Mapped[list["Message"]] = relationship(
        secondary=segmented_memories_message, back_populates="segmented_memories"
    )
    summary = Column(StringEncryptedType(Unicode, encryption_key, AesEngine, "pkcs5"), nullable=False)
    vector_embedding = mapped_column(Vector())
    created_date: Mapped[datetime] = mapped_column(default=datetime.now)
    token_count: Mapped[int]
