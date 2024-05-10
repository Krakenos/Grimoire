from datetime import datetime

from pydantic import BaseModel, ConfigDict, PositiveInt


class KAITokenCount(BaseModel):
    prompt: str


class KAIAbort(BaseModel):
    genkey: str


class Instruct(BaseModel):
    enabled: bool
    collapse_newlines: bool
    system_sequence: str
    system_suffix: str
    input_sequence: str
    input_suffix: str
    output_sequence: str
    output_suffix: str
    first_output_sequence: str
    last_output_sequence: str
    wrap: bool


class Message(BaseModel):
    message: str
    extensionPrompts: list[str]
    injected: bool


class GenerationData(BaseModel):
    description: str
    personality: str
    persona: str
    scenario: str
    char: str
    user: str
    worldInfoBefore: str
    worldInfoAfter: str
    beforeScenarioAnchor: str
    afterScenarioAnchor: str
    storyString: str
    finalMesSend: list[Message]
    main: str
    jailbreak: str
    naiPreamble: str


class Grimoire(BaseModel):
    chat_id: str
    user_id: str = None
    instruct: Instruct | None = None
    generation_data: GenerationData | None = None
    redirect_url: str | None = None


class KAIGeneration(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt: str
    max_context_length: PositiveInt
    max_length: PositiveInt
    grimoire: Grimoire


class OAIGeneration(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt: str
    max_tokens: PositiveInt
    truncation_length: PositiveInt
    stream: bool = False
    api_type: str = "GenericOAI"
    grimoire: Grimoire


class OAITokenize(BaseModel):
    prompt: str


# Tabby endpoint
class OAITokenEncode(BaseModel):
    text: str


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
