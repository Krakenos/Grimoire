from pydantic import BaseModel


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
