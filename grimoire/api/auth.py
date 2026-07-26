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
    mounted sub-apps). Uses PANEL_KEY if set, falling back to AUTH_KEY; if neither is configured
    the panel API is left unauthenticated, matching the panel's existing dev-mode default.
    """
    panel_key = settings.PANEL_KEY or settings.AUTH_KEY
    if not panel_key or authorization == f"Bearer {panel_key}":
        return
    else:
        raise HTTPException(status_code=401, detail="Unauthorized")
