from datetime import datetime

from pydantic import BaseModel


class User(BaseModel):
    id: int
    external_id: str


class Chat(BaseModel):
    id: int
    external_id: str


class ChatMessage(BaseModel):
    id: int
    message_index: int
    message: str
    created_date: datetime


class Knowledge(BaseModel):
    id: int
    entity: str
    summary: str
    updated_date: datetime


class ExternalId(BaseModel):
    external_id: str
