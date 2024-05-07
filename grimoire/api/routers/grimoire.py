from fastapi import APIRouter

router = APIRouter(tags=["Grimoire specific endpoints"])


@router.get("/users")
def get_users():
    pass


@router.get("/users/{user_id}")
def get_user():
    pass


@router.get("/users/{user_id}/chats")
def get_chats():
    pass


@router.get("/users/{user_id}/chats/{chat_id}")
def get_chat():
    pass


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
