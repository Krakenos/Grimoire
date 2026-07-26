import pathlib

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from grimoire.api.routers import panel as panel_router

_STATIC_DIR = pathlib.Path(__file__).parent / "static" / "panel"

panel_app = FastAPI(title="Grimoire Management Panel", docs_url=None, redoc_url=None)
panel_app.include_router(panel_router.router, prefix="/api")
panel_app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="panel-static")
