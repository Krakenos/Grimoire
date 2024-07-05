from pydantic import BaseModel


class Instruct(BaseModel):
    enabled: bool = True
    collapse_newlines: bool = False
    system_sequence: str
    system_suffix: str
    input_sequence: str
    input_suffix: str
    output_sequence: str
    output_suffix: str
    first_output_sequence: str
    last_output_sequence: str
    wrap: bool = False
    trailing_newline: bool = False


class Message(BaseModel):
    message: str
    extensionPrompts: list[str] | None = None
    injected: bool


class AuthorsNote(BaseModel):
    text: str
    depth: int


class GenerationData(BaseModel):
    description: str | None = None
    personality: str | None = None
    persona: str | None = None
    scenario: str | None = None
    char: str
    user: str
    worldInfoBefore: str | None = None
    worldInfoAfter: str | None = None
    beforeScenarioAnchor: str | None = None
    afterScenarioAnchor: str | None = None
    storyString: str | None = None
    finalMesSend: list[Message] | None = None
    main: str | None = None
    jailbreak: str | None = None
    naiPreamble: str | None = None
    authors_note: AuthorsNote | None = None


class Grimoire(BaseModel):
    chat_id: str
    user_id: str | None = None
    instruct: Instruct | None = None
    generation_data: GenerationData | None = None
    redirect_url: str | None = None
    redirect_auth: str | None = None
