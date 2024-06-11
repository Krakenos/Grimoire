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
