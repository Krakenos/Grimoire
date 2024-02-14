import requests
import sseclient
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from memoir.api.request_models import OAIGeneration, OAITokenize
from memoir.common.utils import get_passthrough
from memoir.core.memoir import process_prompt, update_instruct
from memoir.core.settings import settings

router = APIRouter(tags=["Generic OAI passthrough"])


@router.get('/v1/models')
async def get_models(request: Request):
    return get_passthrough('/v1/models', settings['main_api']['auth_key'])


@router.post('/v1/completions')
async def completions(oai_request: OAIGeneration, request: Request):
    def streaming_messages(url, data_json):
        streaming_response = requests.post(url, stream=True,
                                           headers={'Authorization': f'Bearer {settings['main_api']['auth_key']}'},
                                           json=data_json)
        client = sseclient.SSEClient(streaming_response)
        for event in client.events():
            yield f'data: {event.data}\n\n'

    passthrough_json = oai_request.model_dump()
    if oai_request.memoir.instruct is not None:
        update_instruct(oai_request.memoir.instruct)
    passthrough_url = settings['main_api']['url'] + '/v1/completions'
    passthrough_json['api_server'] = settings['main_api']['url'] + '/'
    new_prompt = process_prompt(oai_request.prompt, oai_request.memoir.chat_id, oai_request.truncation_length)
    passthrough_json['prompt'] = new_prompt
    if oai_request.stream:
        return StreamingResponse(streaming_messages(passthrough_url, passthrough_json), media_type="text/event-stream")

    else:
        engine_response = requests.post(passthrough_url,
                                        headers={'Authorization': f'Bearer {settings['main_api']['auth_key']}'},
                                        json=passthrough_json)
        return engine_response.json()


@router.post('/v1/tokenize')
async def tokenize(tokenize_req: OAITokenize):
    passthrough_url = settings['main_api']['url'] + '/v1/tokenize'
    passthrough_json = tokenize_req.model_dump()
    engine_response = requests.post(passthrough_url,
                                    headers={'Authorization': f'Bearer {settings['main_api']['auth_key']}'},
                                    json=passthrough_json)
    return engine_response.json()
