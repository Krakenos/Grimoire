from pydantic import BaseModel, ConfigDict, PositiveInt

from grimoire.api.schemas.passthrough import Grimoire


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


class OAITokenEncode(BaseModel):
    text: str
