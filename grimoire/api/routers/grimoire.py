from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from starlette import status
from starlette.responses import Response

from grimoire.api.schemas.grimoire import (
    ChatData,
    ChatIn,
    ChatMessageIn,
    ChatMessageOut,
    ChatOut,
    ExternalId,
    KnowledgeData,
    KnowledgeDetailOut,
    KnowledgeDetailPatch,
    KnowledgeIn,
    KnowledgeOut,
    MemoriesOut,
    UserIn,
    UserOut, AutoLorebookResponse, AutoLorebookRequest, LorebookStatusRequest, LorebookStatusResponse,
)
from grimoire.common import api_utils
from grimoire.core.grimoire import process_request, generate_lorebook
from grimoire.db.connection import get_db

router = APIRouter(tags=["Grimoire specific endpoints"])


@router.get("/users", response_model=list[UserOut])
def get_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = api_utils.get_users(db, skip, limit)
    return users


@router.post("/users", response_model=UserOut)
def create_user(user: UserIn, db: Session = Depends(get_db)):
    db_user = api_utils.get_user_by_external(db, user.external_id)
    if db_user is not None:
        raise HTTPException(status_code=400, detail="User already exists")
    new_user = api_utils.create_user(db, user.external_id)
    return new_user


@router.post("/users/get_by_external_id", response_model=UserOut)
def get_user_by_external(external_id: ExternalId, db: Session = Depends(get_db)):
    db_user = api_utils.get_user_by_external(db, external_id.external_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@router.get("/users/{user_id}", response_model=UserOut)
def get_user(user_id: int, db: Session = Depends(get_db)):
    db_user = api_utils.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@router.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    db_user = api_utils.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    api_utils.delete_user(db, db_user)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/users/{user_id}/chats", response_model=list[ChatOut])
def get_chats(user_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    chats = api_utils.get_chats(db, user_id=user_id, skip=skip, limit=limit)
    return chats


@router.post("/users/{user_id}/chats/get_by_external_id", response_model=ChatOut)
def get_chat_by_external(user_id: int, external_id: ExternalId, db: Session = Depends(get_db)):
    db_chat = api_utils.get_chat_by_external(db, external_id.external_id, user_id)
    if db_chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return db_chat


@router.get("/users/{user_id}/chats/{chat_id}", response_model=ChatOut)
def get_chat(user_id: int, chat_id: int, db: Session = Depends(get_db)):
    db_chat = api_utils.get_chat(db, user_id=user_id, chat_id=chat_id)
    if db_chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return db_chat


@router.put("/users/{user_id}/chats/{chat_id}", response_model=ChatOut)
def update_chat(chat: ChatIn, user_id: int, chat_id: int, db: Session = Depends(get_db)):
    db_chat = api_utils.get_chat(db, user_id=user_id, chat_id=chat_id)
    if db_chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    db_chat = api_utils.update_record(db, db_chat, chat)
    return db_chat


@router.delete("/users/{user_id}/chats/{chat_id}")
def delete_chat(user_id: int, chat_id: int, db: Session = Depends(get_db)):
    db_chat = api_utils.get_chat(db, user_id=user_id, chat_id=chat_id)
    if db_chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    api_utils.delete_chat(db, db_chat)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/users/{user_id}/chats/{chat_id}/messages", response_model=list[ChatMessageOut])
def get_messages(user_id: int, chat_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    messages = api_utils.get_messages(db, user_id=user_id, chat_id=chat_id, skip=skip, limit=limit)
    return messages


@router.get("/users/{user_id}/chats/{chat_id}/messages/{message_index}", response_model=ChatMessageOut)
def get_message(user_id: int, chat_id: int, message_index: int, db: Session = Depends(get_db)):
    db_message = api_utils.get_message(db, user_id=user_id, chat_id=chat_id, message_index=message_index)
    if db_message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    return db_message


@router.put("/users/{user_id}/chats/{chat_id}/messages/{message_index}", response_model=ChatMessageOut)
def update_message(
    message: ChatMessageIn, user_id: int, chat_id: int, message_index: int, db: Session = Depends(get_db)
):
    db_message = api_utils.get_message(db, user_id=user_id, chat_id=chat_id, message_index=message_index)
    if db_message is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    db_message = api_utils.update_record(db, db_message, message)
    return db_message


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


@router.get("/users/{user_id}/chats/{chat_id}/knowledge", response_model=list[KnowledgeOut])
def get_all_knowledge(user_id: int, chat_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    knowledge = api_utils.get_all_knowledge(db, user_id=user_id, chat_id=chat_id, skip=skip, limit=limit)
    return knowledge


@router.get("/users/{user_id}/chats/{chat_id}/knowledge/{knowledge_id}", response_model=KnowledgeOut)
def get_knowledge(user_id: int, chat_id: int, knowledge_id: int, db: Session = Depends(get_db)):
    db_knowledge = api_utils.get_knowledge(db, user_id=user_id, chat_id=chat_id, knowledge_id=knowledge_id)
    if db_knowledge is None:
        raise HTTPException(status_code=404, detail="Knowledge not found")
    return db_knowledge


@router.put("/users/{user_id}/chats/{chat_id}/knowledge/{knowledge_id}", response_model=KnowledgeOut)
def update_knowledge(
    knowledge: KnowledgeIn, user_id: int, chat_id: int, knowledge_id: int, db: Session = Depends(get_db)
):
    db_knowledge = api_utils.get_knowledge(db, user_id=user_id, chat_id=chat_id, knowledge_id=knowledge_id)
    if db_knowledge is None:
        raise HTTPException(status_code=404, detail="Knowledge not found")
    db_knowledge = api_utils.update_record(db, db_knowledge, knowledge)
    return db_knowledge


@router.patch("/users/{user_id}/chats/{chat_id}/knowledge/{knowledge_id}", response_model=KnowledgeDetailOut)
def patch_knowledge(
    knowledge: KnowledgeDetailPatch, user_id: int, chat_id: int, knowledge_id: int, db: Session = Depends(get_db)
):
    db_knowledge = api_utils.get_knowledge(db, user_id=user_id, chat_id=chat_id, knowledge_id=knowledge_id)

    old_summary = db_knowledge.summary
    new_summary = knowledge.summary

    if db_knowledge is None:
        raise HTTPException(status_code=404, detail="Knowledge not found")

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


@router.get("/users/{user_id}/chats/{chat_id}/memories", response_model=list[MemoriesOut])
def get_all_memories(user_id: int, chat_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    memories = api_utils.get_all_memories(db, user_id=user_id, chat_id=chat_id, skip=skip, limit=limit)
    return memories


@router.get("/users/{user_id}/chats/{chat_id}/memory_graph", response_model=dict)
def get_memory_graph(user_id: int, chat_id: int, db: Session = Depends(get_db)):
    return api_utils.get_memory_graph(db, chat_id, user_id)


@router.post("/get_data", response_model=list[KnowledgeData])
def get_data(chat_data: ChatData, db: Session = Depends(get_db)):
    chat_texts = [message.text for message in chat_data.messages]
    messages_names = [message.sender_name for message in chat_data.messages]
    messages_external_ids = [message.external_id for message in chat_data.messages]
    characters = chat_data.characters
    include_names = chat_data.include_names
    return process_request(
        chat_data.external_chat_id,
        chat_texts,
        messages_external_ids,
        messages_names,
        db,
        characters,
        include_names,
        chat_data.external_user_id,
        chat_data.max_tokens,
    )

@router.post("/autolorebook/create", response_model=AutoLorebookResponse)
def autolorebook_create(req: AutoLorebookRequest):
    request_id = generate_lorebook(req.text)
    return AutoLorebookResponse(request_id=request_id)


@router.post("/autolorebook/status", response_model=LorebookStatusResponse)
def autolorebook_create(req: LorebookStatusRequest):
    return api_utils.get_autolorebook(req.request_id)