from fastapi import APIRouter, Depends, HTTPException
from fastapi.openapi.models import Response
from sqlalchemy.orm import Session

from grimoire.api import grimoire_utils, request_models
from grimoire.db.connection import get_db
from grimoire.db.models import User

router = APIRouter(tags=["Grimoire specific endpoints"])


@router.get("/users", response_model=list[request_models.User])
def get_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = grimoire_utils.get_users(db, skip, limit)
    return users


@router.post("/users", response_model=request_models.User)
def create_user(user: request_models.User, db: Session = Depends(get_db)):
    new_user = User(external_id=user.external_id)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@router.post("/users/get_by_external_id", response_model=request_models.User)
def get_user_by_external(external_id: request_models.ExternalId, db: Session = Depends(get_db)):
    db_user = grimoire_utils.get_user_by_external(db, external_id.external_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@router.get("/users/{user_id}", response_model=request_models.User)
def get_user(user_id: int, db: Session = Depends(get_db)):
    db_user = grimoire_utils.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@router.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    db_user = grimoire_utils.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(db_user)
    db.commit()
    return Response(status_code=204)


@router.get("/users/{user_id}/chats", response_model=list[request_models.Chat])
def get_chats(user_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    chats = grimoire_utils.get_chats(db, user_id=user_id, skip=skip, limit=limit)
    return chats


@router.post("/users/{user_id}/chats/get_by_external_id", response_model=request_models.Chat)
def get_chat_by_external(user_id: int, external_id: request_models.ExternalId, db: Session = Depends(get_db)):
    db_chat = grimoire_utils.get_chat_by_external(db, external_id.external_id, user_id)
    if db_chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return db_chat


@router.get("/users/{user_id}/chats/{chat_id}", response_model=request_models.Chat)
def get_chat(user_id: int, chat_id: int, db: Session = Depends(get_db)):
    db_chat = grimoire_utils.get_chat(db, user_id=user_id, chat_id=chat_id)
    if db_chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return db_chat


@router.put("/users/{user_id}/chats/{chat_id}", response_model=request_models.Chat)
def update_chat(chat: request_models.Chat, user_id: int, chat_id: int, db: Session = Depends(get_db)):
    db_chat = grimoire_utils.get_chat(db, user_id=user_id, chat_id=chat_id)
    new_attributes = chat.model_dump(exclude_unset=True, exclude_none=True, exclude_defaults=True)
    if db_chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    for key, value in new_attributes.items():
        setattr(db_chat, key, value)
    db.commit()
    return db_chat


@router.delete("/users/{user_id}/chats/{chat_id}")
def delete_chat(user_id: int, chat_id: int, db: Session = Depends(get_db)):
    db_chat = grimoire_utils.get_chat(db, user_id=user_id, chat_id=chat_id)
    if db_chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    db.delete(db_chat)
    db.commit()
    return Response(status_code=204)


@router.get("/users/{user_id}/chats/{chat_id}/messages", response_model=list[request_models.ChatMessage])
def get_messages(user_id: int, chat_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    messages = grimoire_utils.get_messages(db, user_id=user_id, chat_id=chat_id, skip=skip, limit=limit)
    return messages


@router.get("/users/{user_id}/chats/{chat_id}/messages/{message_index}", response_model=request_models.ChatMessage)
def get_message(user_id: int, chat_id: int, message_index: int, db: Session = Depends(get_db)):
    db_message = grimoire_utils.get_message(db, user_id=user_id, chat_id=chat_id, message_index=message_index)
    if db_message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    return db_message


@router.put("/users/{user_id}/chats/{chat_id}/messages/{message_id}", response_model=request_models.ChatMessage)
def update_message(
    message: request_models.ChatMessage, user_id: int, chat_id: int, message_index: int, db: Session = Depends(get_db)
):
    db_message = grimoire_utils.get_message(db, user_id=user_id, chat_id=chat_id, message_index=message_index)
    new_attributes = message.model_dump(exclude_unset=True, exclude_none=True, exclude_defaults=True)
    if db_message is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    for key, value in new_attributes.items():
        setattr(db_message, key, value)
    db.commit()
    return db_message


@router.delete("/users/{user_id}/chats/{chat_id}/messages/{message_index}")
def delete_message(user_id: int, chat_id: int, message_index: int, db: Session = Depends(get_db)):
    db_message = grimoire_utils.get_message(db, user_id=user_id, chat_id=chat_id, message_index=message_index)
    if db_message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    db.delete(db_message)
    db.commit()
    return Response(status_code=204)


@router.get("/users/{user_id}/chats/{chat_id}/knowledge", response_model=list[request_models.Knowledge])
def get_all_knowledge(user_id: int, chat_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    knowledge = grimoire_utils.get_all_knowledge(db, user_id=user_id, chat_id=chat_id, skip=skip, limit=limit)
    return knowledge


@router.get("/users/{user_id}/chats/{chat_id}/knowledge/{knowledge_id}", response_model=request_models.Knowledge)
def get_knowledge(user_id: int, chat_id: int, knowledge_id: int, db: Session = Depends(get_db)):
    db_knowledge = grimoire_utils.get_knowledge(db, user_id=user_id, chat_id=chat_id, knowledge_id=knowledge_id)
    if db_knowledge is None:
        raise HTTPException(status_code=404, detail="Knowledge not found")
    return db_knowledge


@router.put("/users/{user_id}/chats/{chat_id}/knowledge/{knowledge_id}", response_model=request_models.Knowledge)
def update_knowledge(
    knowledge: request_models.Knowledge, user_id: int, chat_id: int, knowledge_id: int, db: Session = Depends(get_db)
):
    db_knowledge = grimoire_utils.get_knowledge(db, user_id=user_id, chat_id=chat_id, knowledge_id=knowledge_id)
    new_attributes = knowledge.model_dump(exclude_unset=True, exclude_none=True, exclude_defaults=True)
    if db_knowledge is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    for key, value in new_attributes.items():
        setattr(db_knowledge, key, value)
    db.commit()
    return db_knowledge


@router.delete("/users/{user_id}/chats/{chat_id}/knowledge/{knowledge_id}")
def delete_knowledge(user_id: int, chat_id: int, knowledge_id: int, db: Session = Depends(get_db)):
    db_knowledge = grimoire_utils.get_knowledge(db, user_id=user_id, chat_id=chat_id, knowledge_id=knowledge_id)
    if db_knowledge is None:
        raise HTTPException(status_code=404, detail="Message not found")
    db.delete(db_knowledge)
    db.commit()
    return Response(status_code=204)
