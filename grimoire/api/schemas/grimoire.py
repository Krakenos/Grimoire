from datetime import datetime

from pydantic import BaseModel


class UserOut(BaseModel):
    id: int
    external_id: str


class UserIn(BaseModel):
    external_id: str


class ChatOut(BaseModel):
    id: int
    external_id: str


class ChatIn(BaseModel):
    id: int
    external_id: str


class ChatMessageOut(BaseModel):
    message_index: int
    message: str
    created_date: datetime


class ChatMessageIn(BaseModel):
    message: str
    created_date: datetime


class KnowledgeOut(BaseModel):
    id: int
    entity: str
    entity_label: str
    summary: str
    enabled: bool
    frozen: bool
    updated_date: datetime


class KnowledgeDetailOut(BaseModel):
    entity: str
    entity_label: str
    summary: str
    enabled: bool
    frozen: bool
    updated_date: datetime


class KnowledgeDetailPatch(BaseModel):
    entity: str | None = None
    entity_label: str | None = None
    summary: str | None = None
    enabled: bool | None = None
    frozen: bool | None = None
    updated_date: datetime | None = None


class KnowledgeIn(BaseModel):
    entity: str
    summary: str
    enabled: bool
    frozen: bool
    updated_date: datetime


class ExternalId(BaseModel):
    external_id: str


class ChatDataMessage(BaseModel):
    external_id: str | None = None
    sender_name: str
    text: str


class ChatDataCharacter(BaseModel):
    name: str
    description: str | None = None
    character_note: str | None = None


class ChatDataLorebookEntry(BaseModel):
    keys: list[str]
    description: str


class ChatData(BaseModel):
    external_chat_id: str
    external_user_id: str | None = None
    include_names: bool = True
    max_tokens: int | None = None
    messages: list[ChatDataMessage]
    characters: list[ChatDataCharacter] | None = None
    lorebook_entries: list[ChatDataLorebookEntry] | None = None


class KnowledgeData(BaseModel):
    text: str
    relevance: int


class MemoriesOut(BaseModel):
    id: int
    summary: str
    token_count: int
    created_date: datetime


class MemoriesDetailOut(BaseModel):
    summary: str
    token_count: int
    created_date: datetime


class MemoryPatch(BaseModel):
    summary: str


class GraphNode(BaseModel):
    label: str
    memory_list: list[int]
    id: str


class GraphEdge(BaseModel):
    weight: int
    label: str
    memory_list: list[int]
    source: str
    target: str


class GraphOut(BaseModel):
    directed: bool = False
    multigraph: bool = False
    graph: dict = {}
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class AutoLorebookRequest(BaseModel):
    text: str


class AutoLorebookResponse(BaseModel):
    request_id: str


class LorebookEntry(BaseModel):
    comment: str = "No comment provided"
    key: list[str] = []
    keysecondary: list[str] = []
    content: str = "No content provided"
    position: int = 0
    constant: bool = False
    order: int = 100
    disable: bool = False
    selective: bool = True
    selectiveLogic: int = 0


class Lorebook(BaseModel):
    name: str = "Automatically generated Lorebook"
    description: str = "No description provided"
    entries: dict[str, LorebookEntry] = {}


class LorebookStatusResponse(BaseModel):
    status: str = "processing"
    lorebook: Lorebook | None = None


class LorebookStatusRequest(BaseModel):
    request_id: str


# ---------------------------------------------------------------------------
# Management panel — editable settings
# ---------------------------------------------------------------------------


class SummarizationApiSettingsOut(BaseModel):
    backend: str
    model: str
    url: str
    auth_key_set: bool  # auth_key itself is never sent back to the client
    context_length: int
    system_sequence: str
    system_suffix: str
    input_sequence: str
    input_suffix: str
    output_sequence: str
    output_suffix: str
    first_output_sequence: str
    last_output_sequence: str
    bos_token: str


class SummarizationApiSettingsIn(BaseModel):
    backend: str | None = None
    model: str | None = None
    url: str | None = None
    auth_key: str | None = None  # blank/omitted preserves the currently stored value
    context_length: int | None = None
    system_sequence: str | None = None
    system_suffix: str | None = None
    input_sequence: str | None = None
    input_suffix: str | None = None
    output_sequence: str | None = None
    output_suffix: str | None = None
    first_output_sequence: str | None = None
    last_output_sequence: str | None = None
    bos_token: str | None = None


class SummarizationPromptSettingsOut(BaseModel):
    prompt: str
    segmented_memory_prompt: str
    limit_rate: int
    max_tokens: int
    params: dict


class SummarizationPromptSettingsIn(BaseModel):
    prompt: str | None = None
    segmented_memory_prompt: str | None = None
    limit_rate: int | None = None
    max_tokens: int | None = None
    params: dict | None = None


class TokenizationSettingsOut(BaseModel):
    prefer_local_tokenizer: bool
    local_tokenizer: str


class TokenizationSettingsIn(BaseModel):
    prefer_local_tokenizer: bool | None = None
    local_tokenizer: str | None = None


class PanelSettingsOut(BaseModel):
    summarization_api: SummarizationApiSettingsOut
    summarization: SummarizationPromptSettingsOut
    tokenization: TokenizationSettingsOut
    overridden_sections: list[str]  # sections currently backed by a DB override, vs. file defaults


class PanelSettingsIn(BaseModel):
    summarization_api: SummarizationApiSettingsIn | None = None
    summarization: SummarizationPromptSettingsIn | None = None
    tokenization: TokenizationSettingsIn | None = None


class SettingsResetRequest(BaseModel):
    section: str
