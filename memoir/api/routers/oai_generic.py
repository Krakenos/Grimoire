import requests
import sseclient
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from memoir.api.request_schemas import OAIGenerationInputSchema, OAITokenizeSchema
from memoir.common.utils import get_passthrough
from memoir.core.memoir import process_prompt
from memoir.core.settings import MAIN_API_AUTH, MAIN_API_URL

router = APIRouter(tags=["Generic OAI passthrough"])


@router.get('/v1/models')
async def get_models(request: Request):
    return get_passthrough('/v1/models', MAIN_API_AUTH)


@router.post('/v1/completions')
async def completions(oai_request: OAIGenerationInputSchema, request: Request):
    def streaming_messages(url, data_json):
        streaming_response = requests.post(url, stream=True,
                                           headers={'Authorization': f'Bearer {MAIN_API_AUTH}'}, json=data_json)
        client = sseclient.SSEClient(streaming_response)
        for event in client.events():
            yield f'data: {event.data}\n\n'

    passthrough_json = oai_request.model_dump()
    passthrough_url = MAIN_API_URL + '/v1/completions'
    passthrough_json['api_server'] = MAIN_API_URL + '/'
    new_prompt = process_prompt(oai_request.prompt, oai_request.memoir.chat_id, oai_request.truncation_length)
    passthrough_json['prompt'] = new_prompt
    if oai_request.stream:
        return StreamingResponse(streaming_messages(passthrough_url, passthrough_json), media_type="text/event-stream")

    else:
        engine_response = requests.post(passthrough_url, headers={'Authorization': f'Bearer {MAIN_API_AUTH}'},
                                        json=passthrough_json)
        return engine_response.json()


@router.post('/v1/tokenize')
async def tokenize(tokenize_req: OAITokenizeSchema):
    passthrough_url = MAIN_API_URL + '/v1/tokenize'
    passthrough_json = tokenize_req.model_dump()
    engine_response = requests.post(passthrough_url, headers={'Authorization': f'Bearer {MAIN_API_AUTH}'},
                                    json=passthrough_json)
    return engine_response.json()
