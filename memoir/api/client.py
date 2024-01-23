from fastapi import FastAPI

from memoir.api.routers import kobold, oai_generic

app = FastAPI()
app.include_router(kobold.router)
app.include_router(oai_generic.router)
