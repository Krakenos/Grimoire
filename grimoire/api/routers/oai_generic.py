from urllib.parse import urljoin

import requests
import sseclient
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from grimoire.api.request_models import OAIGeneration, OAITokenize, OAITokenEncode
from grimoire.common.utils import get_passthrough
from grimoire.core.grimoire import process_prompt, update_instruct
from grimoire.core.settings import settings

router = APIRouter(tags=["Generic OAI passthrough"])


@router.get('/v1/models')
@router.get('/v1/model/list')
async def get_models(request: Request):
    return get_passthrough('/v1/models', settings['main_api']['auth_key'])


@router.get('/v1/model')
async def model():
    return get_passthrough('/v1/model', settings['main_api']['auth_key'])


@router.post('/v1/completions')
async def completions(oai_request: OAIGeneration, request: Request):
    def streaming_messages(url, data_json):
        streaming_response = requests.post(url, stream=True,
                                           headers={'Authorization': f"Bearer {settings['main_api']['auth_key']}"},
                                           json=data_json)
        client = sseclient.SSEClient(streaming_response)
        for event in client.events():
            yield f'data: {event.data}\n\n'

    passthrough_json = oai_request.model_dump()
    if oai_request.grimoire.instruct is not None:
        update_instruct(oai_request.grimoire.instruct)
    passthrough_url = urljoin(settings['main_api']['url'], '/v1/completions')
    passthrough_json['api_server'] = settings['main_api']['url']
    new_prompt = process_prompt(oai_request.prompt, oai_request.grimoire.chat_id, oai_request.truncation_length,
                                oai_request.api_type, oai_request.grimoire.generation_data)
    passthrough_json['prompt'] = new_prompt
    if oai_request.stream:
        return StreamingResponse(streaming_messages(passthrough_url, passthrough_json), media_type="text/event-stream")

    else:
        engine_response = requests.post(passthrough_url,
                                        headers={'Authorization': f"Bearer {settings['main_api']['auth_key']}"},
                                        json=passthrough_json)
        return engine_response.json()


@router.post('/v1/tokenize')
async def tokenize(tokenize_req: OAITokenize):
    passthrough_url = urljoin(settings['main_api']['url'], '/v1/tokenize')
    passthrough_json = tokenize_req.model_dump()
    engine_response = requests.post(passthrough_url,
                                    headers={'Authorization': f"Bearer {settings['main_api']['auth_key']}"},
                                    json=passthrough_json)
    return engine_response.json()


@router.post('/v1/token/encode')
async def token_encode(tokenize_req: OAITokenEncode):
    passthrough_url = urljoin(settings['main_api']['url'], '/v1/token_encode')
    passthrough_json = tokenize_req.model_dump()
    engine_response = requests.post(passthrough_url,
                                    headers={'Authorization': f"Bearer {settings['main_api']['auth_key']}"},
                                    json=passthrough_json)
    return engine_response.json()
