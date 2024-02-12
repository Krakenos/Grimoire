from celery import Celery
from celery_singleton import Singleton
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from memoir.common.llm_helpers import count_context, generate_text
from memoir.common.loggers import summary_logger
from memoir.core.settings import SIDE_API_URL, DB_ENGINE, CELERY_BROKER_URL, SUMMARIZATION_PARAMS, \
    SUMMARIZATION_PROMPT, SUMMARIZATION_START_TOKEN, SUMMARIZATION_INPUT_SEQ, SUMMARIZATION_OUTPUT_SEQ, \
    SINGLE_API_MODE, MAIN_API_URL, MAIN_API_BACKEND, MAIN_API_AUTH, MODEL_INPUT_SEQUENCE, MODEL_OUTPUT_SEQUENCE, \
    SIDE_API_AUTH, SIDE_API_BACKEND
from memoir.db.models import Knowledge, Message

celery_app = Celery('tasks', broker=CELERY_BROKER_URL)


def make_summary_prompt(session, knowledge_entry, max_context: int) -> str | None:
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
    chat_id = knowledge_entry.chat_id
    if knowledge_entry.summary:
        summary = knowledge_entry.summary
    else:
        summary = ''
    message_indices = [message.message_index for message in knowledge_entry.messages]
    chunk_indices = set()
    for message_index in message_indices:
        chunk_indices.update([message_index - 1, message_index, message_index + 1])
    chunk_indices -= {-1, 0}
    if len(chunk_indices) < 2:
        return None
    final_indices = sorted(list(chunk_indices))
    query = select(Message.message).where(Message.message_index.in_(final_indices),
                                          Message.chat_id == chat_id).order_by(Message.message_index)
    query_results = session.execute(query).all()
    messages = [row[0] for row in query_results]
    prompt = ''
    reversed_messages = []
    for message in messages[::-1]:
        reversed_messages.append(message)
        messages_text = '\n'.join(reversed_messages[::-1])
        new_prompt = SUMMARIZATION_PROMPT.format(term=knowledge_entry.entity,
                                                 previous_summary=summary,
                                                 messages=messages_text,
                                                 start_token=SUMMARIZATION_START_TOKEN,
                                                 input_sequence=input_sequence,
                                                 output_sequence=output_sequence)
        new_tokens = count_context(new_prompt, summarization_backend, summarization_url, summarization_auth)
        if new_tokens > max_context:
            break
        else:
            prompt = new_prompt
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
        prompt = make_summary_prompt(session, knowledge_entry, context_len)
        if prompt is None:  # Edge case of having 1 message for summary, only may happen at start of chat
            return None
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
