import json
import os
import re
import sqlite3

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
db = create_engine('sqlite:///db.sqlite3')


def make_summary_prompt(session, term):
    prompt = f'<|system|>Based on following text summarize what or who {term} is. Keep explanation short\n'
    instance = session.query(Knowledge).filter_by(entity=term).first()
    for message in instance.messages:
        prompt += message.message + '\n'
    prompt += '<|model|>'
    return prompt


def summarize(session, term):
    prompt = make_summary_prompt(session, term)
    json = {'prompt': prompt, 'max_length': 350}
    kobold_response = requests.post(KOBOLD_URL + '/api/v1/generate', json=json)
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
    messages = re.split(r"<\|user\|>|<\|model\|>", prompt)
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
