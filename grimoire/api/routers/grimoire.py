from fastapi import APIRouter, Depends, HTTPException
from fastapi.openapi.models import Response
from sqlalchemy.orm import Session

from grimoire.api.schemas.grimoire import Chat, ChatMessage, ExternalId, Knowledge, User
from grimoire.common import api_utils
from grimoire.db.connection import get_db

router = APIRouter(tags=["Grimoire specific endpoints"])


@router.get("/users", response_model=list[User])
def get_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = api_utils.get_users(db, skip, limit)
    return users


@router.post("/users", response_model=User)
def create_user(user: User, db: Session = Depends(get_db)):
    new_user = api_utils.create_user(db, user.external_id)
    return new_user


@router.post("/users/get_by_external_id", response_model=User)
def get_user_by_external(external_id: ExternalId, db: Session = Depends(get_db)):
    db_user = api_utils.get_user_by_external(db, external_id.external_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@router.get("/users/{user_id}", response_model=User)
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
    db.delete(db_user)
    db.commit()
    return Response(status_code=204)


@router.get("/users/{user_id}/chats", response_model=list[Chat])
def get_chats(user_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    chats = api_utils.get_chats(db, user_id=user_id, skip=skip, limit=limit)
    return chats


@router.post("/users/{user_id}/chats/get_by_external_id", response_model=Chat)
def get_chat_by_external(user_id: int, external_id: ExternalId, db: Session = Depends(get_db)):
    db_chat = api_utils.get_chat_by_external(db, external_id.external_id, user_id)
    if db_chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return db_chat


@router.get("/users/{user_id}/chats/{chat_id}", response_model=Chat)
def get_chat(user_id: int, chat_id: int, db: Session = Depends(get_db)):
    db_chat = api_utils.get_chat(db, user_id=user_id, chat_id=chat_id)
    if db_chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return db_chat


@router.put("/users/{user_id}/chats/{chat_id}", response_model=Chat)
def update_chat(chat: Chat, user_id: int, chat_id: int, db: Session = Depends(get_db)):
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
    db.delete(db_chat)
    db.commit()
    return Response(status_code=204)


@router.get("/users/{user_id}/chats/{chat_id}/messages", response_model=list[ChatMessage])
def get_messages(user_id: int, chat_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    messages = api_utils.get_messages(db, user_id=user_id, chat_id=chat_id, skip=skip, limit=limit)
    return messages


@router.get("/users/{user_id}/chats/{chat_id}/messages/{message_index}", response_model=ChatMessage)
def get_message(user_id: int, chat_id: int, message_index: int, db: Session = Depends(get_db)):
    db_message = api_utils.get_message(db, user_id=user_id, chat_id=chat_id, message_index=message_index)
    if db_message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    return db_message


@router.put("/users/{user_id}/chats/{chat_id}/messages/{message_id}", response_model=ChatMessage)
def update_message(message: ChatMessage, user_id: int, chat_id: int, message_index: int, db: Session = Depends(get_db)):
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
    db.delete(db_message)
    db.commit()
    return Response(status_code=204)


@router.get("/users/{user_id}/chats/{chat_id}/knowledge", response_model=list[Knowledge])
def get_all_knowledge(user_id: int, chat_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    knowledge = api_utils.get_all_knowledge(db, user_id=user_id, chat_id=chat_id, skip=skip, limit=limit)
    return knowledge


@router.get("/users/{user_id}/chats/{chat_id}/knowledge/{knowledge_id}", response_model=Knowledge)
def get_knowledge(user_id: int, chat_id: int, knowledge_id: int, db: Session = Depends(get_db)):
    db_knowledge = api_utils.get_knowledge(db, user_id=user_id, chat_id=chat_id, knowledge_id=knowledge_id)
    if db_knowledge is None:
        raise HTTPException(status_code=404, detail="Knowledge not found")
    return db_knowledge


@router.put("/users/{user_id}/chats/{chat_id}/knowledge/{knowledge_id}", response_model=Knowledge)
def update_knowledge(
    knowledge: Knowledge, user_id: int, chat_id: int, knowledge_id: int, db: Session = Depends(get_db)
):
    db_knowledge = api_utils.get_knowledge(db, user_id=user_id, chat_id=chat_id, knowledge_id=knowledge_id)
    if db_knowledge is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    db_knowledge = api_utils.update_record(db, db_knowledge, knowledge)
    return db_knowledge


@router.delete("/users/{user_id}/chats/{chat_id}/knowledge/{knowledge_id}")
def delete_knowledge(user_id: int, chat_id: int, knowledge_id: int, db: Session = Depends(get_db)):
    db_knowledge = api_utils.get_knowledge(db, user_id=user_id, chat_id=chat_id, knowledge_id=knowledge_id)
    if db_knowledge is None:
        raise HTTPException(status_code=404, detail="Message not found")
    db.delete(db_knowledge)
    db.commit()
    return Response(status_code=204)
