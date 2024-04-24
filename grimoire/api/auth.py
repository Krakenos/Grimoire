from typing import Annotated

from fastapi import Header, HTTPException

from grimoire.core.settings import settings


def check_api_key(authorization: Annotated[str | None, Header()]) -> None:
    grimoire_api_key = settings["AUTH_KEY"]
    if not grimoire_api_key or authorization == f"Bearer {grimoire_api_key}":
        return
    else:
        raise HTTPException(status_code=401, detail="Unauthorized")
