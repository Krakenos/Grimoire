import copy
from urllib.parse import urljoin

import requests
import sseclient
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from grimoire.api.schemas.oai_passthrough import OAIGeneration, OAITokenEncode, OAITokenize
from grimoire.common.utils import get_passthrough
from grimoire.core.grimoire import process_prompt, update_instruct
from grimoire.core.settings import settings
from grimoire.db.connection import get_db

router = APIRouter(tags=["Generic OAI passthrough"])


@router.get("/v1/models")
@router.get("/v1/model/list")
async def get_models(request: Request):
    return get_passthrough("/v1/models", settings["main_api"]["auth_key"])


@router.get("/v1/model")
async def model():
    return get_passthrough("/v1/model", settings["main_api"]["auth_key"])


@router.post("/v1/completions")
async def completions(oai_request: OAIGeneration, request: Request, db: Session = Depends(get_db)):
    def streaming_messages(url, auth_key, data_json):
        streaming_response = requests.post(
            url, stream=True, headers={"Authorization": f"Bearer {auth_key}"}, json=data_json
        )
        client = sseclient.SSEClient(streaming_response)
        for event in client.events():
            yield f"data: {event.data}\n\n"

    passthrough_json = oai_request.model_dump()

    current_settings = copy.deepcopy(settings)

    if oai_request.grimoire.instruct is not None:
        current_settings = update_instruct(oai_request.grimoire.instruct)

    if oai_request.grimoire.redirect_url is not None:
        current_settings["main_api"]["url"] = oai_request.grimoire.redirect_url

    if oai_request.grimoire.redirect_auth is not None:
        current_settings["main_api"]["auth_key"] = oai_request.grimoire.redirect_auth

    passthrough_url = urljoin(current_settings["main_api"]["url"], "/v1/completions")
    auth_key = current_settings["main_api"]["auth_key"]
    passthrough_json["api_server"] = current_settings["main_api"]["url"]

    max_context = oai_request.truncation_length - oai_request.max_tokens
    new_prompt = await process_prompt(
        prompt=oai_request.prompt,
        chat_id=oai_request.grimoire.chat_id,
        context_length=max_context,
        db_session=db,
        api_type=oai_request.api_type,
        generation_data=oai_request.grimoire.generation_data,
        user_id=oai_request.grimoire.user_id,
        current_settings=current_settings,
    )

    passthrough_json["prompt"] = new_prompt

    if oai_request.stream:
        return StreamingResponse(
            streaming_messages(passthrough_url, auth_key, passthrough_json), media_type="text/event-stream"
        )

    else:
        engine_response = requests.post(
            passthrough_url, headers={"Authorization": f"Bearer {auth_key}"}, json=passthrough_json
        )
        return engine_response.json()


@router.post("/v1/tokenize")
async def tokenize(tokenize_req: OAITokenize):
    passthrough_url = urljoin(settings["main_api"]["url"], "/v1/tokenize")
    passthrough_json = tokenize_req.model_dump()
    engine_response = requests.post(
        passthrough_url, headers={"Authorization": f"Bearer {settings['main_api']['auth_key']}"}, json=passthrough_json
    )
    return engine_response.json()


@router.post("/v1/token/encode")
async def token_encode(tokenize_req: OAITokenEncode):
    passthrough_url = urljoin(settings["main_api"]["url"], "/v1/token_encode")
    passthrough_json = tokenize_req.model_dump()
    engine_response = requests.post(
        passthrough_url, headers={"Authorization": f"Bearer {settings['main_api']['auth_key']}"}, json=passthrough_json
    )
    return engine_response.json()
