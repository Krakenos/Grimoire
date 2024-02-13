from typing import Optional

from pydantic import BaseModel, PositiveInt, ConfigDict


class KAITokenCountSchema(BaseModel):
    prompt: str


class KAIAbortSchema(BaseModel):
    genkey: str


class InstructSchema(BaseModel):
    enabled: bool
    system_prompt: str
    input_sequence: str
    output_sequence: str
    first_output_sequence: str
    last_output_sequence: str
    system_sequence_prefix: str
    system_sequence_suffix: str
    stop_sequence: str
    separator_sequence: str
    wrap: bool
    names: bool


class MemoirSchema(BaseModel):
    chat_id: str
    instruct: Optional[InstructSchema] = None


class KAIGenerationInputSchema(BaseModel):
    model_config = ConfigDict(extra='allow')

    prompt: str
    max_context_length: PositiveInt
    max_length: PositiveInt
    memoir: MemoirSchema


class OAIGenerationInputSchema(BaseModel):
    model_config = ConfigDict(extra='allow')

    prompt: str
    max_tokens: PositiveInt
    truncation_length: PositiveInt
    stream: bool = False
    memoir: MemoirSchema


class OAITokenizeSchema(BaseModel):
    prompt: str
