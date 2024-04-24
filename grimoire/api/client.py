from fastapi import Depends, FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from grimoire.api.auth import check_api_key
from grimoire.api.routers import kobold, oai_generic
from grimoire.common.loggers import general_logger

app = FastAPI(dependencies=[Depends(check_api_key)])
app.include_router(kobold.router)
app.include_router(oai_generic.router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    general_logger.debug(f"{request}: {exc_str}")
    content = {"status_code": 422, "message": exc_str, "data": None}
    return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)
