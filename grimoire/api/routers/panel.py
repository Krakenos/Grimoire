from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import ValidationError
from sqlalchemy.orm import Session
from starlette import status
from starlette.responses import Response

from grimoire.api.schemas.grimoire import (
    AutoLorebookRequest,
    AutoLorebookResponse,
    ChatMessageIn,
    ChatMessageOut,
    ChatOut,
    GraphOut,
    KnowledgeDetailPatch,
    KnowledgeOut,
    LorebookStatusRequest,
    LorebookStatusResponse,
    MemoriesOut,
    MemoryPatch,
    PanelSettingsIn,
    PanelSettingsOut,
    SettingsResetRequest,
    SummarizationApiSettingsOut,
    SummarizationPromptSettingsOut,
    TokenizationSettingsOut,
    UserOut,
)
from grimoire.common import api_utils
from grimoire.core.grimoire import extract_text_from_upload, generate_lorebook
from grimoire.core.runtime_settings import (
    EDITABLE_SECTIONS,
    delete_override,
    get_effective_settings,
    get_overridden_sections,
    upsert_override,
)
from grimoire.core.settings import ApiSettings, SummarizationSettings, TokenizationSettings
from grimoire.db.connection import get_db

router = APIRouter(tags=["Management panel"])


# ---------------------------------------------------------------------------
# Lorebook generation (chat-independent)
# ---------------------------------------------------------------------------


@router.post("/autolorebook/create", response_model=AutoLorebookResponse)
def lorebook_create(req: AutoLorebookRequest):
    request_id = str(generate_lorebook(req.text))
    return AutoLorebookResponse(request_id=request_id)


@router.post("/autolorebook/upload_file", response_model=AutoLorebookResponse)
async def lorebook_create_from_file(file: UploadFile):
    text = await extract_text_from_upload(file)
    request_id = str(generate_lorebook(text))
    return AutoLorebookResponse(request_id=request_id)


@router.post("/autolorebook/status", response_model=LorebookStatusResponse)
def lorebook_status(req: LorebookStatusRequest):
    return api_utils.get_autolorebook(req.request_id)


# ---------------------------------------------------------------------------
# Users + Chats (read-only — navigation for the panel sidebar)
# ---------------------------------------------------------------------------


@router.get("/users", response_model=list[UserOut])
def list_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return api_utils.get_users(db, skip=skip, limit=limit)


@router.get("/users/{user_id}/chats", response_model=list[ChatOut])
def list_chats(user_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    user = api_utils.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return api_utils.get_chats(db, user_id=user_id, skip=skip, limit=limit)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@router.get("/users/{user_id}/chats/{chat_id}/messages", response_model=list[ChatMessageOut])
def list_messages(user_id: int, chat_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    chat = api_utils.get_chat(db, user_id=user_id, chat_id=chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return api_utils.get_messages(db, user_id=user_id, chat_id=chat_id, skip=skip, limit=limit)


@router.patch("/users/{user_id}/chats/{chat_id}/messages/{message_index}", response_model=ChatMessageOut)
def update_message(
    user_id: int, chat_id: int, message_index: int, message: ChatMessageIn, db: Session = Depends(get_db)
):
    db_message = api_utils.get_message(db, user_id=user_id, chat_id=chat_id, message_index=message_index)
    if db_message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    return api_utils.update_record(db, db_message, message)


@router.delete("/users/{user_id}/chats/{chat_id}/messages/{message_index}")
def delete_message(user_id: int, chat_id: int, message_index: int, db: Session = Depends(get_db)):
    db_message = api_utils.get_message(db, user_id=user_id, chat_id=chat_id, message_index=message_index)
    if db_message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    for named_entity in db_message.spacy_named_entities:
        db.delete(named_entity)
    db.delete(db_message)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Knowledge
# ---------------------------------------------------------------------------


@router.get("/users/{user_id}/chats/{chat_id}/knowledge", response_model=list[KnowledgeOut])
def list_knowledge(user_id: int, chat_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    chat = api_utils.get_chat(db, user_id=user_id, chat_id=chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return api_utils.get_all_knowledge(db, user_id=user_id, chat_id=chat_id, skip=skip, limit=limit)


@router.patch("/users/{user_id}/chats/{chat_id}/knowledge/{knowledge_id}", response_model=KnowledgeOut)
def update_knowledge(
    user_id: int, chat_id: int, knowledge_id: int, knowledge: KnowledgeDetailPatch, db: Session = Depends(get_db)
):
    db_knowledge = api_utils.get_knowledge(db, user_id=user_id, chat_id=chat_id, knowledge_id=knowledge_id)
    if db_knowledge is None:
        raise HTTPException(status_code=404, detail="Knowledge not found")

    old_summary = db_knowledge.summary
    new_summary = knowledge.summary

    db_knowledge = api_utils.update_record(db, db_knowledge, knowledge)

    if new_summary and new_summary != old_summary:
        db_knowledge = api_utils.update_summary_metadata(db, db_knowledge)

    return db_knowledge


@router.delete("/users/{user_id}/chats/{chat_id}/knowledge/{knowledge_id}")
def delete_knowledge(user_id: int, chat_id: int, knowledge_id: int, db: Session = Depends(get_db)):
    db_knowledge = api_utils.get_knowledge(db, user_id=user_id, chat_id=chat_id, knowledge_id=knowledge_id)
    if db_knowledge is None:
        raise HTTPException(status_code=404, detail="Knowledge not found")
    db.delete(db_knowledge)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Segmented memories
# ---------------------------------------------------------------------------


@router.get("/users/{user_id}/chats/{chat_id}/memories", response_model=list[MemoriesOut])
def list_memories(user_id: int, chat_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    chat = api_utils.get_chat(db, user_id=user_id, chat_id=chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return api_utils.get_all_memories(db, user_id=user_id, chat_id=chat_id, skip=skip, limit=limit)


@router.patch("/users/{user_id}/chats/{chat_id}/memories/{memory_id}", response_model=MemoriesOut)
def update_memory(user_id: int, chat_id: int, memory_id: int, body: MemoryPatch, db: Session = Depends(get_db)):
    db_memory = api_utils.get_memory(db, user_id=user_id, chat_id=chat_id, memory_id=memory_id)
    if db_memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")

    old_summary = db_memory.summary
    db_memory.summary = body.summary
    if body.summary != old_summary:
        db_memory = api_utils.update_memory_metadata(db, db_memory)
    else:
        db.add(db_memory)
        db.commit()
        db.refresh(db_memory)

    return db_memory


@router.delete("/users/{user_id}/chats/{chat_id}/memories/{memory_id}")
def delete_memory(user_id: int, chat_id: int, memory_id: int, db: Session = Depends(get_db)):
    db_memory = api_utils.get_memory(db, user_id=user_id, chat_id=chat_id, memory_id=memory_id)
    if db_memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    api_utils.delete_memory(db, db_memory)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Memory graph (view-only)
# ---------------------------------------------------------------------------


@router.get("/users/{user_id}/chats/{chat_id}/memory_graph", response_model=GraphOut)
def get_memory_graph(user_id: int, chat_id: int, db: Session = Depends(get_db)):
    chat = api_utils.get_chat(db, user_id=user_id, chat_id=chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return api_utils.get_memory_graph(db, chat_id=chat_id, user_id=user_id)


# ---------------------------------------------------------------------------
# Settings (summarization API, prompts, sampler params, tokenization)
# ---------------------------------------------------------------------------


def _build_settings_out(db: Session) -> PanelSettingsOut:
    runtime_settings = get_effective_settings(db)
    api_settings = runtime_settings.summarization_api
    summarization_settings = runtime_settings.summarization
    tokenization_settings = runtime_settings.tokenization

    return PanelSettingsOut(
        summarization_api=SummarizationApiSettingsOut(
            backend=api_settings.backend,
            model=api_settings.model,
            url=api_settings.url,
            auth_key_set=bool(api_settings.auth_key),
            context_length=api_settings.context_length,
            system_sequence=api_settings.system_sequence,
            system_suffix=api_settings.system_suffix,
            input_sequence=api_settings.input_sequence,
            input_suffix=api_settings.input_suffix,
            output_sequence=api_settings.output_sequence,
            output_suffix=api_settings.output_suffix,
            first_output_sequence=api_settings.first_output_sequence,
            last_output_sequence=api_settings.last_output_sequence,
            bos_token=api_settings.bos_token,
        ),
        summarization=SummarizationPromptSettingsOut(
            prompt=summarization_settings.prompt,
            segmented_memory_prompt=summarization_settings.segmented_memory_prompt,
            limit_rate=summarization_settings.limit_rate,
            max_tokens=summarization_settings.max_tokens,
            params=summarization_settings.params,
        ),
        tokenization=TokenizationSettingsOut(
            prefer_local_tokenizer=tokenization_settings.prefer_local_tokenizer,
            local_tokenizer=tokenization_settings.local_tokenizer,
        ),
        overridden_sections=sorted(get_overridden_sections(db)),
    )


@router.get("/settings", response_model=PanelSettingsOut)
def get_panel_settings(db: Session = Depends(get_db)):
    return _build_settings_out(db)


@router.put("/settings", response_model=PanelSettingsOut)
def update_panel_settings(body: PanelSettingsIn, db: Session = Depends(get_db)):
    current = get_effective_settings(db)

    if body.summarization_api is not None:
        merged = current.summarization_api.model_dump()
        updates = body.summarization_api.model_dump(exclude_unset=True, exclude_none=True)
        if not updates.get("auth_key"):  # blank/omitted auth_key means "leave unchanged"
            updates.pop("auth_key", None)
        merged.update(updates)
        try:
            ApiSettings(**merged)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors()) from e
        upsert_override(db, "summarization_api", merged)

    if body.summarization is not None:
        merged = current.summarization.model_dump()
        updates = body.summarization.model_dump(exclude_unset=True, exclude_none=True)
        merged.update(updates)
        try:
            SummarizationSettings(**merged)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors()) from e
        upsert_override(db, "summarization", merged)

    if body.tokenization is not None:
        merged = current.tokenization.model_dump()
        updates = body.tokenization.model_dump(exclude_unset=True, exclude_none=True)
        merged.update(updates)
        try:
            TokenizationSettings(**merged)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors()) from e
        upsert_override(db, "tokenization", merged)

    return _build_settings_out(db)


@router.post("/settings/reset", response_model=PanelSettingsOut)
def reset_panel_settings(body: SettingsResetRequest, db: Session = Depends(get_db)):
    if body.section not in EDITABLE_SECTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown settings section: {body.section}")
    delete_override(db, body.section)
    return _build_settings_out(db)
