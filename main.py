import requests
import sseclient
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from memoir.api.request_schemas import KAITokenCountSchema, OAIGenerationInputSchema, KAIGenerationInputSchema
from memoir.memoir import process_prompt
from memoir.settings import MAIN_API_URL, MAIN_API_AUTH
from memoir.utils import get_passthrough

app = FastAPI()


# KAI Endpoint
@app.get('/api/v1/model')
async def model():
    return get_passthrough('/api/v1/model')


# KAI Endpoint
@app.get('/api/v1/info/version')
async def info_version():
    return get_passthrough('/api/v1/info/version')


# KAI Endpoint
@app.get('/api/extra/version')
async def extra_version():
    return get_passthrough('/api/extra/version')


# KAI Endpoint
@app.post('/api/v1/generate')
async def generate(k_request: KAIGenerationInputSchema):
    passthrough_json = k_request.model_dump()
    new_prompt = process_prompt(k_request.prompt, k_request.memoir.chat_id, k_request.max_context_length)
    passthrough_url = MAIN_API_URL + '/api/v1/generate'
    passthrough_json['prompt'] = new_prompt
    kobold_response = requests.post(passthrough_url, json=passthrough_json)
    return kobold_response.json()


# KAI Endpoint
@app.post('/api/extra/tokencount')
async def token_count(k_request: KAITokenCountSchema):
    passthrough_json = k_request.model_dump(exclude_defaults=True)
    passthrough_url = MAIN_API_URL + '/api/extra/tokencount'
    kobold_response = requests.post(passthrough_url, json=passthrough_json)
    return kobold_response.json()


# Aphrodite/Ooba/OAI endpoint
@app.get('/v1/models')
async def get_models(request: Request):
    return get_passthrough('/v1/models', MAIN_API_AUTH)


@app.post('/v1/completions')
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


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=5555)
