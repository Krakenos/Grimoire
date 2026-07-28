from typing import Annotated

from fastapi import Header, HTTPException

from grimoire.core.settings import settings


def check_api_key(authorization: Annotated[str | None, Header()] = None) -> None:
    grimoire_api_key = settings.AUTH_KEY
    if not grimoire_api_key or authorization == f"Bearer {grimoire_api_key}":
        return
    else:
        raise HTTPException(status_code=401, detail="Unauthorized")


def check_panel_key(authorization: Annotated[str | None, Header()] = None) -> None:
    """Gate for the management panel API (/panel/api/...).

    The panel is mounted as a separate sub-application, so the main app's `check_api_key`
    dependency does not apply to it (Starlette doesn't propagate parent-app dependencies to
    mounted sub-apps).

    The panel is a local administration tool: it reads and writes every user's chats, so the
    supported way to run in production is with enable_management_panel off entirely. PANEL_KEY is
    therefore optional, and an unset key leaves the panel open — appropriate when it is only
    reachable from your own machine, and no substitute for not exposing it in the first place.

    AUTH_KEY is deliberately not accepted as a fallback: it is distributed to every chat client,
    so honouring it would silently make each client key a key over every other user's data.
    """
    panel_key = settings.PANEL_KEY
    if not panel_key or authorization == f"Bearer {panel_key}":
        return
    else:
        raise HTTPException(status_code=401, detail="Unauthorized")
