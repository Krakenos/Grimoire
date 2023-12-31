import re

import requests
import spacy
import sseclient
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from transformers import AutoTokenizer

from loggers import general_logger, summary_logger, context_logger
from models import Knowledge, Message
from request_schemas import KAITokenCountSchema, OAIGenerationInputSchema
from settings import SIDE_API_URL, MAIN_API_URL, CONTEXT_PERCENTAGE, DB_ENGINE, MAIN_API_AUTH, MODEL_INPUT_SEQUENCE, \
    MODEL_OUTPUT_SEQUENCE, MAIN_API_BACKEND

app = FastAPI()
nlp = spacy.load("en_core_web_trf")
db = create_engine(DB_ENGINE)


def make_summary_prompt(session, term):
    prompt = f'<|system|>Based on following text summarize what or who {term} is. Keep explanation short<|user|>'
    instance = session.query(Knowledge).filter_by(entity=term).first()
    for message in instance.messages:
        prompt += message.message + '\n'
    prompt += '<|model|>'
    return prompt


def summarize(session, term):
    prompt = make_summary_prompt(session, term)
    json = {'prompt': prompt, 'max_length': 350}
    kobold_response = requests.post(SIDE_API_URL + '/api/v1/generate', json=json)
    summary_logger.debug(kobold_response)


def orm_get_or_create(session, db_model, **kwargs):
    instance = session.query(db_model).filter_by(**kwargs).first()
    if instance:
        return instance
    else:
        instance = db_model(**kwargs)
        session.add(instance)
        session.commit()
        return instance


def process_prompt(prompt):
    context_logger.debug(prompt)
    banned_labels = ['DATE', 'CARDINAL', 'ORDINAL']
    pattern = re.escape(MODEL_INPUT_SEQUENCE) + r'|' + re.escape(MODEL_OUTPUT_SEQUENCE)
    messages = re.split(pattern, prompt)
    last_messages = messages[-3:-1]
    docs = list(nlp.pipe(last_messages))
    with Session(db) as session:
        for doc in docs:
            context_logger.debug(doc.text_with_ws)
            db_message = orm_get_or_create(session, Message, message=doc.text)
            ent_list = [(str(ent), ent.label_) for ent in doc.ents if ent.label_ not in banned_labels]
            unique_ents = list(set(ent_list))
            for ent, ent_label in unique_ents:
                knowledge_entity = orm_get_or_create(session, Knowledge, entity=ent, entity_type='NAMED ENTITY',
                                                     entity_label=ent_label)
                knowledge_entity.messages.append(db_message)
                session.add(knowledge_entity)
                session.commit()
            # for entity in doc.ents:
            #     context_logger.debug(entity.text, entity.label_, spacy.explain(entity.label_))
            #     summarize(session, entity.text)


def count_context(text):
    if MAIN_API_BACKEND == 'Aphrodite':
        models_endpoint = MAIN_API_URL + '/v1/models'
        response = requests.get(models_endpoint, headers={'Authorization': f'Bearer {MAIN_API_AUTH}'})
        model_name = response.json()['data'][0]['id']
        # Default to llama tokenizer if model tokenizer is not on huggingface
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except OSError as s:
            general_logger.warning('Could not load model tokenizer, defaulting to llama-tokenizer')
            general_logger.warning(s)
            tokenizer = AutoTokenizer.from_pretrained('oobabooga/llama-tokenizer')
        encoded = tokenizer(text)
        token_amount = len(encoded['input_ids'])
        return token_amount

    else:
        token_count_endpoint = MAIN_API_URL + '/api/extra/tokencount'
        request_body = {'prompt': text}
        kobold_response = requests.post(token_count_endpoint, json=request_body)
        value = int(kobold_response.json()['value'])
        return value


def get_context_length(api_url: str) -> int:
    length_endpoint = api_url + '/v1/config/max_context_length'
    kobold_response = requests.get(length_endpoint)
    value = int(kobold_response.json()['value'])
    return value


def fill_context(prompt, context_size):
    max_context = context_size
    max_memoir_context = max_context * CONTEXT_PERCENTAGE
    banned_labels = ['DATE', 'CARDINAL', 'ORDINAL']
    pattern = re.escape(MODEL_INPUT_SEQUENCE) + r'|' + re.escape(MODEL_OUTPUT_SEQUENCE)
    messages = re.split(pattern, prompt)
    prompt_definitions = messages[0]  # first portion should always be instruction and char definitions
    docs = list(nlp.pipe(messages))
    full_ent_list = []
    for doc in docs:
        general_logger.debug(doc.text_with_ws)
        ent_list = [(str(ent), ent.label_) for ent in doc.ents if ent.label_ not in banned_labels]
        full_ent_list += ent_list
    unique_ents = list(dict.fromkeys(full_ent_list[::-1]))  # unique ents ordered from bottom of context
    summaries = []
    with Session(db) as session:
        for ent in unique_ents:
            instances = session.query(Knowledge).filter(Knowledge.entity == ent[0],
                                                        Knowledge.entity_type == 'NAMED ENTITY',
                                                        Knowledge.entity_label == ent[1], Knowledge.summary.isnot(None),
                                                        Knowledge.summary.isnot(''), Knowledge.token_count.isnot(None),
                                                        Knowledge.token_count.isnot(0))
            instance = instances.first()
            summaries.append((instance.summary, instance.token_count))
    memoir_token_sum = sum([summary_tuple[1] for summary_tuple in summaries])
    while memoir_token_sum > max_memoir_context:
        summaries.pop()
        memoir_token_sum = sum([summary_tuple[1] for summary_tuple in summaries])
    memoir_text = ''
    for summary in summaries:
        memoir_text = memoir_text + summary[0] + '\n'
    memoir_text_len = count_context(memoir_text)
    definitions_context_len = count_context(prompt_definitions)
    max_chat_context = max_context - definitions_context_len - memoir_text_len
    starting_message = 1
    messages_text = '\n'.join(messages[starting_message:])
    messages_len = count_context(messages_text)
    while messages_len > max_chat_context:
        starting_message += 1
        messages_text = '\n'.join(messages[starting_message:])
        messages_len = count_context(messages_text)
    final_prompt = '\n'.join([prompt_definitions, memoir_text, messages_text])
    return final_prompt


def get_passthrough(endpoint: str, auth_token=None) -> dict:
    passthrough_url = MAIN_API_URL + endpoint
    kobold_response = requests.get(passthrough_url, headers={'Authorization': f'Bearer {auth_token}'})
    return kobold_response.json()


def get_model_name():
    if MAIN_API_BACKEND == 'Aphrodite':
        pass
    else:
        pass


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
async def generate(k_request: Request):
    passthrough_json = await k_request.json()
    process_prompt(passthrough_json['prompt'])
    passthrough_url = MAIN_API_URL + '/api/v1/generate'
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

    if oai_request.stream:
        return StreamingResponse(streaming_messages(passthrough_url, passthrough_json), media_type="text/event-stream")

    else:
        engine_response = requests.post(passthrough_url, headers={'Authorization': f'Bearer {MAIN_API_AUTH}'},
                                        json=passthrough_json)
        return engine_response.json()


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=5555)
