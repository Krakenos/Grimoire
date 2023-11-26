import json
import re

import requests
import spacy
import uvicorn
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from models import Knowledge, Message
from request_schemas import KAIGenerationInputSchema, KAITokenCountSchema

app = FastAPI()
nlp = spacy.load("en_core_web_trf")
KOBOLD_URL = 'http://127.0.0.1:5001'
SIDE_API_URL = 'http://127.0.0.1:5002'
CONTEXT_PERCENTAGE = 0.25
db = create_engine('sqlite:///db.sqlite3')


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
    print(kobold_response)


def orm_get_or_create(session, model, **kwargs):
    instance = session.query(model).filter_by(**kwargs).first()
    if instance:
        return instance
    else:
        instance = model(**kwargs)
        session.add(instance)
        session.commit()
        return instance


def process_prompt(prompt):
    print(prompt)
    banned_labels = ['DATE', 'CARDINAL', 'ORDINAL']
    messages = re.split(r"<\|user\|>|<\|model\|>", prompt)  # TODO make the instruct mode in/out changeable
    last_messages = messages[-3:-1]
    docs = list(nlp.pipe(last_messages))
    with Session(db) as session:
        for doc in docs:
            print(doc.text_with_ws)
            db_message = orm_get_or_create(session, Message, message=doc.text)
            ent_list = [(str(ent), ent.label_) for ent in doc.ents if ent.label_ not in banned_labels]
            unique_ents = list(set(ent_list))
            for ent, ent_label in unique_ents:
                knowledge_entity = orm_get_or_create(session, Knowledge, entity=ent, entity_type='NAMED ENTITY',
                                                     entity_label=ent_label)
                knowledge_entity.messages.append(db_message)
                session.add(knowledge_entity)
                session.commit()
            for entity in doc.ents:
                print(entity.text, entity.label_, spacy.explain(entity.label_))
                summarize(session, entity.text)


def count_context(text):
    token_count_endpoint = KOBOLD_URL + '/api/extra/tokencount'
    request_body = {'prompt': text}
    kobold_response = requests.post(token_count_endpoint, json=request_body)
    value = int(kobold_response.json()['value'])
    return value


def get_context_length(api_url: str) -> int:
    length_endpoint = api_url + '/v1/config/max_context_length'
    kobold_response = requests.get(length_endpoint)
    value = int(kobold_response.json()['value'])
    return value


def fill_context(prompt):
    max_context = get_context_length(KOBOLD_URL)
    max_memoir_context = max_context * CONTEXT_PERCENTAGE
    banned_labels = ['DATE', 'CARDINAL', 'ORDINAL']
    messages = re.split(r"<\|user\|>|<\|model\|>", prompt)  # TODO make the instruct mode in/out changeable
    prompt_definitions = messages[0]  # first portion should always be instruction and char definitions
    docs = list(nlp.pipe(messages))
    full_ent_list = []
    for doc in docs:
        print(doc.text_with_ws)
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


def get_passthrough(endpoint: str) -> dict:
    passthrough_url = KOBOLD_URL + endpoint
    kobold_response = requests.get(passthrough_url)
    return json.loads(kobold_response.text)


@app.get('/api/v1/model')
async def model():
    return get_passthrough('/api/v1/model')


@app.get('/api/v1/info/version')
async def info_version():
    return get_passthrough('/api/v1/info/version')


@app.get('/api/extra/version')
async def extra_version():
    return get_passthrough('/api/extra/version')


@app.post('/api/v1/generate')
async def generate(k_request: KAIGenerationInputSchema):
    passthrough_json = k_request.model_dump(exclude_defaults=True)
    process_prompt(passthrough_json['prompt'])
    passthrough_url = KOBOLD_URL + '/api/v1/generate'
    kobold_response = requests.post(passthrough_url, json=passthrough_json)
    return kobold_response.content


@app.post('/api/extra/tokencount')
async def token_count(k_request: KAITokenCountSchema):
    passthrough_json = k_request.model_dump(exclude_defaults=True)
    passthrough_url = KOBOLD_URL + '/api/extra/tokencount'
    kobold_response = requests.post(passthrough_url, json=passthrough_json)
    return json.loads(kobold_response.text)


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=5555)
