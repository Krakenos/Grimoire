import requests
from celery import Celery

from models import Knowledge
from settings import SIDE_API_URL

celery_app = Celery('tasks', broker='amqp://guest@localhost//')


def make_summary_prompt(session, term):
    prompt = f'<|system|>Based on following text summarize what or who {term} is. Keep explanation short<|user|>'
    instance = session.query(Knowledge).filter_by(entity=term).first()
    for message in instance.messages:
        prompt += message.message + '\n'
    prompt += '<|model|>'
    return prompt


@celery_app.task
def summarize(session, term):
    prompt = make_summary_prompt(session, term)
    json = {'prompt': prompt, 'max_length': 350}
    kobold_response = requests.post(SIDE_API_URL + '/api/v1/generate', json=json)
    print(kobold_response)
