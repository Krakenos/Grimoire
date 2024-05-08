from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from grimoire.api import grimoire_utils, request_models
from grimoire.db.connection import get_db

router = APIRouter(tags=["Grimoire specific endpoints"])


@router.get("/users", response_model=list[request_models.User])
def get_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = grimoire_utils.get_users(db, skip, limit)
    return users


@router.get("/users/{user_id}", response_model=request_models.User)
def get_user(user_id: int, db: Session = Depends(get_db)):
    db_user = grimoire_utils.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@router.get("/users/{user_id}/chats", response_model=list[request_models.Chat])
def get_chats(user_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    chats = grimoire_utils.get_chats(db, user_id=user_id, skip=skip, limit=limit)
    return chats


@router.get("/users/{user_id}/chats/{chat_id}", response_model=request_models.Chat)
def get_chat(user_id: int, chat_id: int, db: Session = Depends(get_db)):
    db_chat = grimoire_utils.get_chat(db, user_id=user_id, chat_id=chat_id)
    if db_chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return db_chat


@router.put("/users/{user_id}/chats/{chat_id}")
def update_chat():
    pass


@router.get("/users/{user_id}/chats/{chat_id}/messages")
def get_messages():
    pass


@router.get("/users/{user_id}/chats/{chat_id}/messages/{message_id}")
def get_message():
    pass


@router.put("/users/{user_id}/chats/{chat_id}/messages/{message_id}")
def update_message():
    pass


@router.get("/users/{user_id}/chats/{chat_id}/knowledge")
def get_all_knowledge():
    pass


@router.get("/users/{user_id}/chats/{chat_id}/knowledge/{knowledge_id}")
def get_knowledge():
    pass


@router.put("/users/{user_id}/chats/{chat_id}/knowledge/{knowledge_id}")
def update_knowledge():
    pass
