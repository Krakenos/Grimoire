from typing import Optional, List

from pydantic import BaseModel, conint, PositiveInt, confloat, NonNegativeInt, NonNegativeFloat, conlist, ConfigDict


class KAITokenCountSchema(BaseModel):
    prompt: str


class KAIGenerationInputSchema(BaseModel):
    model_config = ConfigDict(extra='allow')

    prompt: str
    max_context_length: PositiveInt
    max_length: PositiveInt


class OAIGenerationInputSchema(BaseModel):
    model_config = ConfigDict(extra='allow')

    prompt: str
    truncation_length: PositiveInt
    stream: bool = False
