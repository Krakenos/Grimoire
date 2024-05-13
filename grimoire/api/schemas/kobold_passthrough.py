from pydantic import BaseModel, ConfigDict, PositiveInt

from grimoire.api.schemas.passthrough import Grimoire


class KAITokenCount(BaseModel):
    prompt: str


class KAIAbort(BaseModel):
    genkey: str


class KAIGeneration(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt: str
    max_context_length: PositiveInt
    max_length: PositiveInt
    grimoire: Grimoire
