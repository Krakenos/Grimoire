from fastapi import Depends, FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from grimoire.api.auth import check_api_key
from grimoire.api.routers import grimoire
from grimoire.common.loggers import general_logger
from grimoire.core.settings import settings

app = FastAPI(dependencies=[Depends(check_api_key)])
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    # Authorization and Content-Type must be allowed for the ST extension fetch calls
    allow_headers=["*"],
)
app.include_router(grimoire.router, prefix="/grimoire")

if settings.enable_management_panel:
    from grimoire.api.panel import panel_app  # noqa: PLC0415

    app.mount("/panel", panel_app)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    general_logger.debug(f"{request}: {exc_str}")
    content = {"status_code": 422, "message": exc_str, "data": None}
    return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)
