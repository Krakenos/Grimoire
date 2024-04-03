from celery import Celery
from celery_singleton import Singleton
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from grimoire.common.llm_helpers import count_context, generate_text
from grimoire.common.loggers import summary_logger
from grimoire.core.settings import settings
from grimoire.db.models import Knowledge, Message

celery_app = Celery('tasks', broker=settings['CELERY_BROKER_URL'])


def make_summary_prompt(session, knowledge_entry, max_context: int, api_settings, summarization_settings) -> str | None:
    summarization_url = api_settings['url']
    summarization_backend = api_settings['backend']
    summarization_auth = api_settings['auth_key']
    input_sequence = api_settings['input_sequence']
    output_sequence = api_settings['output_sequence']
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
        new_prompt = summarization_settings['prompt'].format(term=knowledge_entry.entity,
                                                             previous_summary=summary,
                                                             messages=messages_text,
                                                             bos_token=settings['summarization']['bos_token'],
                                                             input_sequence=input_sequence,
                                                             output_sequence=output_sequence)
        new_tokens = count_context(new_prompt, summarization_backend, summarization_url, summarization_auth)
        if new_tokens > max_context:
            break
        else:
            prompt = new_prompt
    return prompt


@celery_app.task(base=Singleton, lock_expiry=60)
def summarize(term: str, label: str, chat_id: int, api_settings: dict = None, summarization_settings: dict = None,
              db_engine: str = None, context_len: int = 4096, response_len: int = 300) -> None:
    db = create_engine(db_engine)
    summarization_url = api_settings['url']
    summarization_backend = api_settings['backend']
    summarization_auth = api_settings['auth_key']
    limit_rate = summarization_settings['limit_rate']
    with Session(db) as session:
        knowledge_entry = session.query(Knowledge).filter(Knowledge.entity.ilike(term),
                                                          Knowledge.entity_type == 'NAMED ENTITY',
                                                          Knowledge.chat_id == chat_id).first()

        if knowledge_entry.update_count < limit_rate:  # Don't summarize if it's below the limit
            return None

        prompt = make_summary_prompt(session, knowledge_entry, context_len, api_settings, summarization_settings)
        if prompt is None:  # Edge case of having 1 message for summary, only may happen at start of chat
            return None
        generation_params = {
            'max_length': response_len,
            'max_tokens': response_len,
            'truncation_length': context_len,
            "max_context_length": context_len,
        }
        generation_params.update(summarization_settings['params'])
        additional_stops = [api_settings['input_sequence'].strip(),
                            api_settings['output_sequence'].strip(),
                            api_settings['first_output_sequence'].strip(),
                            api_settings['last_output_sequence'].strip(),
                            api_settings['separator_sequence'].strip()]
        additional_stops = [stop for stop in additional_stops if stop]
        generation_params['stop'].extend(additional_stops)
        generation_params['stop_sequence'].extend(additional_stops)
        summary_text, request_json = generate_text(prompt,
                                                   generation_params,
                                                   summarization_backend,
                                                   summarization_url,
                                                   summarization_auth)
        summary_text = summary_text.replace('\n\n', '\n')
        knowledge_entry.summary = summary_text
        knowledge_entry.token_count = count_context(summary_text, summarization_backend, summarization_url,
                                                    summarization_auth)
        knowledge_entry.update_count = 1
        summary_logger.debug(f'({knowledge_entry.token_count} tokens){term} ({label}): {summary_text}\n{request_json}')
        session.commit()
