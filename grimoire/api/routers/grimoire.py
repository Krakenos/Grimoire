from fastapi import APIRouter

router = APIRouter(tags=["Grimoire specific endpoints"])


@router.get("/chats")
def get_chats():
    pass


@router.put("/chats")
def update_chats():
    pass


@router.get("/messages")
def get_messages():
    pass


@router.put("/messages")
def update_messages():
    pass


@router.get("/knowledge")
def get_knowledge():
    pass


@router.put("/knowledge")
def update_knowledge():
    pass
