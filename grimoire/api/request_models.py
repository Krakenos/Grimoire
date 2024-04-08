from typing import Optional

from pydantic import BaseModel, PositiveInt, ConfigDict


class KAITokenCount(BaseModel):
    prompt: str


class KAIAbort(BaseModel):
    genkey: str


class Instruct(BaseModel):
    enabled: bool
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
    instruct: Optional[Instruct] = None
    generation_data: Optional[GenerationData] = None


class KAIGeneration(BaseModel):
    model_config = ConfigDict(extra='allow')

    prompt: str
    max_context_length: PositiveInt
    max_length: PositiveInt
    grimoire: Grimoire


class OAIGeneration(BaseModel):
    model_config = ConfigDict(extra='allow')

    prompt: str
    max_tokens: PositiveInt
    truncation_length: PositiveInt
    stream: bool = False
    api_type: str = 'GenericOAI'
    grimoire: Grimoire


class OAITokenize(BaseModel):
    prompt: str


# Tabby endpoint
class OAITokenEncode(BaseModel):
    text: str
