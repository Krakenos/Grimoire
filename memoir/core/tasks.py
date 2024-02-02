import requests
from celery import Celery
from celery_singleton import Singleton
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from memoir.common.llm_helpers import count_context
from memoir.common.loggers import summary_logger
from memoir.core.settings import SIDE_API_URL, DB_ENGINE, CELERY_BROKER_URL, SIDE_API_BACKEND, SUMMARIZATION_PARAMS, \
    SUMMARIZATION_PROMPT, SUMMARIZATION_START_TOKEN, SUMMARIZATION_INPUT_SEQ, SUMMARIZATION_OUTPUT_SEQ
from memoir.db.models import Knowledge

celery_app = Celery('tasks', broker=CELERY_BROKER_URL)


def make_summary_prompt(session: Session, term: str, label: str, chat_id: str, max_context: int) -> str:
    main_prompt = SUMMARIZATION_PROMPT.format(term=term)
    prompt = f'{SUMMARIZATION_START_TOKEN}{SUMMARIZATION_INPUT_SEQ} {main_prompt}\n\n'
    instance = session.query(Knowledge).filter_by(entity=term, chat_id=chat_id, entity_label=label).scalar()
    if instance.summary is not None:
        summary = instance.summary + '\n'
    else:
        summary = ''
    for message in instance.messages[::-1]:  # reverse order to start from latest message
        new_prompt = prompt + message.message + '\n'
        new_prompt += summary
        new_tokens = count_context(new_prompt + SUMMARIZATION_OUTPUT_SEQ, 'KoboldAI', SIDE_API_URL)
        if new_tokens >= max_context:
            break
        else:
            prompt = new_prompt
    prompt += SUMMARIZATION_OUTPUT_SEQ
    return prompt


@celery_app.task(base=Singleton)
def summarize(term: str, label: str, chat_id: str, context_len: int = 4096,
              response_len: int = 300) -> None:
    db = create_engine(DB_ENGINE)
    with Session(db) as session:
        knowledge_entry = session.query(Knowledge).filter(Knowledge.entity.ilike(term),
                                                          Knowledge.entity_type == 'NAMED ENTITY',
                                                          Knowledge.chat_id == chat_id).first()
        prompt = make_summary_prompt(session, knowledge_entry.entity, label, chat_id, context_len)
        json = {
            'prompt': prompt,
            'max_length': response_len,
            "max_context_length": context_len,
        }
        json.update(SUMMARIZATION_PARAMS)
        kobold_response = requests.post(SIDE_API_URL + '/api/v1/generate', json=json)
        response = kobold_response.json()
        summary_text = response['results'][0]['text']
        knowledge_entry.summary = summary_text
        knowledge_entry.token_count = count_context(summary_text, SIDE_API_BACKEND, SIDE_API_URL)
        summary_logger.debug(f'({knowledge_entry.token_count} tokens){term} ({label}): {summary_text}\n{json}')
        session.commit()
