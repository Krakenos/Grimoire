from datetime import datetime

from pydantic import BaseModel


class UserOut(BaseModel):
    id: int
    external_id: str


class UserIn(BaseModel):
    external_id: str


class ChatOut(BaseModel):
    id: int
    external_id: str


class ChatIn(BaseModel):
    id: int
    external_id: str


class ChatMessageOut(BaseModel):
    message_index: int
    message: str
    created_date: datetime


class ChatMessageIn(BaseModel):
    message: str
    created_date: datetime


class KnowledgeOut(BaseModel):
    id: int
    entity: str
    summary: str
    updated_date: datetime


class KnowledgeIn(BaseModel):
    entity: str
    summary: str
    updated_date: datetime


class ExternalId(BaseModel):
    external_id: str


class ChatDataMessage(BaseModel):
    external_id: str | None = None
    sender_name: str
    text: str


class ChatData(BaseModel):
    external_chat_id: str
    external_user_id: str | None = None
    max_tokens: int | None = None
    messages: list[ChatDataMessage]


class KnowledgeData(BaseModel):
    text: str
    relevance: int
