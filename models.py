from datetime import datetime
from typing import Optional, List

from sqlalchemy import Table, Column, ForeignKey, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


knowledge_message = Table(
    'knowledge_message',
    Base.metadata,
    Column('knowledge_id', ForeignKey('knowledge.id'), primary_key=True),
    Column('message_id', ForeignKey('message.id'), primary_key=True),
)


class Knowledge(Base):
    __tablename__ = 'knowledge'

    id: Mapped[int] = mapped_column(primary_key=True)
    entity: Mapped[str]
    entity_type: Mapped[Optional[str]]
    entity_label: Mapped[Optional[str]]
    summary: Mapped[Optional[str]]
    token_count: Mapped[Optional[int]]
    messages: Mapped[List['Message']] = relationship(secondary=knowledge_message)
    updated_date: Mapped[datetime] = mapped_column(default=datetime.now)
    update_at: Mapped[Optional[int]] = mapped_column(default=1)
    update_count: Mapped[Optional[int]] = mapped_column(default=1)

    def __repr__(self):
        return f'id: {self.id}, name: {self.entity}'


class Message(Base):
    __tablename__ = 'message'

    id: Mapped[int] = mapped_column(primary_key=True)
    summary: Mapped[Optional[str]]
    message: Mapped[str]
    summary_tokens: Mapped[Optional[int]]
    message_tokens: Mapped[Optional[int]]
    created_date: Mapped[datetime] = mapped_column(default=datetime.now)
