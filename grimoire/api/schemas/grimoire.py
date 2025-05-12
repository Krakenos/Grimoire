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
    entity_label: str
    summary: str
    enabled: bool
    frozen: bool
    updated_date: datetime


class KnowledgeDetailOut(BaseModel):
    entity: str
    entity_label: str
    summary: str
    enabled: bool
    frozen: bool
    updated_date: datetime


class KnowledgeDetailPatch(BaseModel):
    entity: str | None = None
    entity_label: str | None = None
    summary: str | None = None
    enabled: bool | None = None
    frozen: bool | None = None
    updated_date: datetime | None = None


class KnowledgeIn(BaseModel):
    entity: str
    summary: str
    enabled: bool
    frozen: bool
    updated_date: datetime


class ExternalId(BaseModel):
    external_id: str


class ChatDataMessage(BaseModel):
    external_id: str | None = None
    sender_name: str
    text: str


class ChatDataCharacter(BaseModel):
    name: str
    description: str | None = None
    character_note: str | None = None


class ChatData(BaseModel):
    external_chat_id: str
    external_user_id: str | None = None
    include_names: bool = True
    max_tokens: int | None = None
    messages: list[ChatDataMessage]
    characters: list[ChatDataCharacter] | None = None


class KnowledgeData(BaseModel):
    text: str
    relevance: int


class MemoriesOut(BaseModel):
    id: int
    summary: str
    token_count: int
    created_date: datetime


class MemoriesDetailOut(BaseModel):
    summary: str
    token_count: int
    created_date: datetime

class AutoLorebookRequest(BaseModel):
    text: str

class AutoLorebookResponse(BaseModel):
    request_id: str
