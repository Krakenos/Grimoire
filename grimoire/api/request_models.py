from typing import Optional

from pydantic import BaseModel, PositiveInt, ConfigDict


class KAITokenCount(BaseModel):
    prompt: str


class KAIAbort(BaseModel):
    genkey: str


class Instruct(BaseModel):
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


class Messages(BaseModel):
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
    finalMesSend: list[Messages]
    main: str
    jailbreak: str
    naiPreamble: str


class Grimoire(BaseModel):
    chat_id: str
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
