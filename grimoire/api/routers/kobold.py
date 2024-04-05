import copy
from urllib.parse import urljoin

import requests
from fastapi import APIRouter

from grimoire.api.request_models import KAIGeneration, KAITokenCount, KAIAbort
from grimoire.common.utils import get_passthrough
from grimoire.core.grimoire import process_prompt, update_instruct
from grimoire.core.settings import settings

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
async def generate(k_request: KAIGeneration):
    passthrough_json = k_request.model_dump()
    current_settings = copy.deepcopy(settings)
    if k_request.grimoire.instruct is not None:
        current_settings = update_instruct(k_request.grimoire.instruct)
    new_prompt = process_prompt(prompt=k_request.prompt,
                                chat_id=k_request.grimoire.chat_id,
                                context_length=k_request.max_context_length,
                                api_type='kobold',
                                generation_data=k_request.grimoire.generation_data,
                                user_id=k_request.grimoire.user_id,
                                current_settings=current_settings)
    passthrough_url = urljoin(current_settings['main_api']['url'], '/api/v1/generate')
    passthrough_json['prompt'] = new_prompt
    kobold_response = requests.post(passthrough_url, json=passthrough_json)
    return kobold_response.json()


@router.post('/api/extra/tokencount')
async def token_count(k_request: KAITokenCount):
    passthrough_json = k_request.model_dump(exclude_defaults=True)
    passthrough_url = urljoin(settings['main_api']['url'], '/api/extra/tokencount')
    kobold_response = requests.post(passthrough_url, json=passthrough_json)
    return kobold_response.json()


@router.post('/api/extra/abort')
async def abort(k_request: KAIAbort):
    passthrough_json = k_request.model_dump(exclude_defaults=True)
    passthrough_url = urljoin(settings['main_api']['url'], '/api/extra/abort')
    kobold_response = requests.post(passthrough_url, json=passthrough_json)
    return kobold_response.json()
