import requests
from fastapi import APIRouter

from memoir.api.request_schemas import KAIGenerationInputSchema, KAITokenCountSchema, KAIAbortSchema
from memoir.common.utils import get_passthrough
from memoir.core.memoir import process_prompt
from memoir.core.settings import settings

router = APIRouter(tags=["Kobold passthrough"])


@router.get('/api/v1/model')
async def model():
    return get_passthrough('/api/v1/model')


@router.get('/api/v1/info/version')
async def info_version():
    return get_passthrough('/api/v1/info/version')


@router.get('/api/extra/version')
async def extra_version():
    return get_passthrough('/api/extra/version')


@router.post('/api/v1/generate')
async def generate(k_request: KAIGenerationInputSchema):
    passthrough_json = k_request.model_dump()
    new_prompt = process_prompt(k_request.prompt, k_request.memoir.chat_id, k_request.max_context_length)
    passthrough_url = settings['main_api']['url'] + '/api/v1/generate'
    passthrough_json['prompt'] = new_prompt
    kobold_response = requests.post(passthrough_url, json=passthrough_json)
    return kobold_response.json()


@router.post('/api/extra/tokencount')
async def token_count(k_request: KAITokenCountSchema):
    passthrough_json = k_request.model_dump(exclude_defaults=True)
    passthrough_url = settings['main_api']['url'] + '/api/extra/tokencount'
    kobold_response = requests.post(passthrough_url, json=passthrough_json)
    return kobold_response.json()


@router.post('/api/extra/abort')
async def abort(k_request: KAIAbortSchema):
    passthrough_json = k_request.model_dump(exclude_defaults=True)
    passthrough_url = settings['main_api']['url'] + '/api/extra/abort'
    kobold_response = requests.post(passthrough_url, json=passthrough_json)
    return kobold_response.json()
