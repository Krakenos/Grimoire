from typing import Optional, List

from pydantic import BaseModel, conint, PositiveInt, confloat, NonNegativeInt, NonNegativeFloat, conlist


class KAITokenCountSchema(BaseModel):
    prompt: str


class KAIGenerationInputSchema(BaseModel):
    prompt: str
    n: Optional[conint(ge=1, le=5)] = 1
    max_context_length: PositiveInt
    max_length: PositiveInt
    rep_pen: Optional[confloat(ge=1)] = 1.0
    rep_pen_range: Optional[NonNegativeInt]
    rep_pen_slope: Optional[NonNegativeFloat]
    top_k: Optional[NonNegativeInt] = 0.0
    top_a: Optional[NonNegativeFloat] = 0.0
    top_p: Optional[confloat(ge=0, le=1)] = 1.0
    tfs: Optional[confloat(ge=0, le=1)] = 1.0
    eps_cutoff: Optional[confloat(ge=0, le=1000)] = 0.0
    eta_cutoff: Optional[NonNegativeFloat] = 0.0
    typical: Optional[confloat(ge=0, le=1)] = 1.0
    temperature: Optional[NonNegativeFloat] = 1.0
    use_memory: Optional[bool] = None
    use_story: Optional[bool] = None
    use_authors_note: Optional[bool] = None
    use_world_info: Optional[bool] = None
    use_userscripts: Optional[bool] = None
    soft_prompt: Optional[str] = None
    disable_output_formatting: Optional[bool] = None
    frmtrmblln: Optional[bool] = None
    frmtrmspch: Optional[bool] = None
    singleline: Optional[bool] = None
    use_default_badwordsids: Optional[bool] = None
    disable_input_formatting: Optional[bool] = None
    frmtadsnsp: Optional[bool] = None
    quiet: Optional[bool] = None
    sampler_order: Optional[conlist(int)] = None
    sampler_seed: Optional[conint(ge=0, le=2 ** 64 - 1)] = None
    sampler_full_determinism: Optional[bool] = None
    stop_sequence: Optional[List[str]] = None
    mirostat: Optional[int] = None
    mirostat_eta: Optional[float] = None
    mirostat_tau: Optional[float] = None
    grammar: Optional[str] = None
