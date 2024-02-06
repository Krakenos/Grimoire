from celery import Celery
from celery_singleton import Singleton
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from memoir.common.llm_helpers import count_context, generate_text
from memoir.common.loggers import summary_logger
from memoir.core.settings import SIDE_API_URL, DB_ENGINE, CELERY_BROKER_URL, SUMMARIZATION_PARAMS, \
    SUMMARIZATION_PROMPT, SUMMARIZATION_START_TOKEN, SUMMARIZATION_INPUT_SEQ, SUMMARIZATION_OUTPUT_SEQ, \
    SINGLE_API_MODE, MAIN_API_URL, MAIN_API_BACKEND, MAIN_API_AUTH, MODEL_INPUT_SEQUENCE, MODEL_OUTPUT_SEQUENCE, \
    SIDE_API_AUTH, SIDE_API_BACKEND
from memoir.db.models import Knowledge

celery_app = Celery('tasks', broker=CELERY_BROKER_URL)


def make_summary_prompt(knowledge_entry, max_context: int) -> str:
    if SINGLE_API_MODE:
        summarization_url = MAIN_API_URL
        summarization_backend = MAIN_API_BACKEND
        summarization_auth = MAIN_API_AUTH
        input_sequence = MODEL_INPUT_SEQUENCE
        output_sequence = MODEL_OUTPUT_SEQUENCE
    else:
        summarization_url = SIDE_API_URL
        summarization_backend = SIDE_API_BACKEND
        summarization_auth = SIDE_API_AUTH
        input_sequence = SUMMARIZATION_INPUT_SEQ
        output_sequence = SUMMARIZATION_OUTPUT_SEQ
    instruction_prompt = SUMMARIZATION_PROMPT.format(term=knowledge_entry.entity)
    instruction_prompt = f'{input_sequence}{instruction_prompt}{output_sequence}'
    prompt = f'{SUMMARIZATION_START_TOKEN}'
    if knowledge_entry.summary:
        summary = knowledge_entry.summary + '\n'
    else:
        summary = ''
    prompt = prompt + summary + '\n'
    for message in knowledge_entry.messages[::-1]:  # reverse order to start from latest message
        new_prompt = prompt + message.message + '\n'
        new_tokens = count_context(new_prompt + instruction_prompt,
                                   summarization_backend,
                                   summarization_url,
                                   summarization_auth)
        if new_tokens >= max_context:
            break
        else:
            prompt = new_prompt
    prompt += instruction_prompt
    return prompt


@celery_app.task(base=Singleton)
def summarize(term: str, label: str, chat_id: str, context_len: int = 4096,
              response_len: int = 300) -> None:
    db = create_engine(DB_ENGINE)
    if SINGLE_API_MODE:
        summarization_url = MAIN_API_URL
        summarization_backend = MAIN_API_BACKEND
        summarization_auth = MAIN_API_AUTH
    else:
        summarization_url = SIDE_API_URL
        summarization_backend = SIDE_API_BACKEND
        summarization_auth = SIDE_API_AUTH
    with Session(db) as session:
        knowledge_entry = session.query(Knowledge).filter(Knowledge.entity.ilike(term),
                                                          Knowledge.entity_type == 'NAMED ENTITY',
                                                          Knowledge.chat_id == chat_id).first()
        prompt = make_summary_prompt(knowledge_entry, context_len)
        generation_params = {
            'max_length': response_len,
            'max_tokens': response_len,
            'truncation_length': context_len,
            "max_context_length": context_len,
        }
        generation_params.update(SUMMARIZATION_PARAMS)
        summary_text, request_json = generate_text(prompt,
                                                   generation_params,
                                                   summarization_backend,
                                                   summarization_url,
                                                   summarization_auth)
        knowledge_entry.summary = summary_text
        knowledge_entry.token_count = count_context(summary_text, summarization_backend, summarization_url)
        summary_logger.debug(f'({knowledge_entry.token_count} tokens){term} ({label}): {summary_text}\n{request_json}')
        session.commit()
