import pathlib

from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles

from grimoire.api.auth import check_panel_key
from grimoire.api.routers import panel as panel_router

_STATIC_DIR = pathlib.Path(__file__).parent / "static" / "panel"

panel_app = FastAPI(title="Grimoire Management Panel", docs_url=None, redoc_url=None)
# Gated separately from the static SPA below: the panel is mounted as a sub-app, so it doesn't
# inherit the main app's check_api_key dependency (see check_panel_key's docstring).
panel_app.include_router(panel_router.router, prefix="/api", dependencies=[Depends(check_panel_key)])
panel_app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="panel-static")
