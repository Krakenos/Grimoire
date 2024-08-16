from celery import Celery
from celery_singleton import Singleton
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from grimoire.common.llm_helpers import count_context, generate_text
from grimoire.common.loggers import general_logger, summary_logger
from grimoire.core.settings import settings
from grimoire.db.models import Message
from grimoire.db.queries import get_knowledge_entity
from grimoire.db.secondary_database import get_messages_from_external_db

celery_app = Celery("tasks", broker=settings["CELERY_BROKER_URL"])


def make_summary_prompt(
    session,
    knowledge_entry,
    max_context: int,
    api_settings,
    summarization_settings,
    secondary_database_settings,
    include_names: bool = True,
) -> str | None:
    summarization_url = api_settings["url"]
    summarization_backend = api_settings["backend"]
    summarization_auth = api_settings["auth_key"]
    input_sequence = api_settings["input_sequence"]
    input_suffix = api_settings["input_suffix"]
    output_sequence = api_settings["output_sequence"]
    secondary_database = secondary_database_settings["enabled"]
    secondary_database_url = secondary_database_settings["db_engine"]
    secondary_database_encryption_method = secondary_database_settings["message_encryption"]
    secondary_database_encryption_key = secondary_database_settings["encryption_key"]
    chat_id = knowledge_entry.chat_id

    if knowledge_entry.summary:
        summary = knowledge_entry.summary
    else:
        summary = ""

    message_indices = [message.message_index for message in knowledge_entry.messages]
    chunk_indices = set()
    for message_index in message_indices:
        chunk_indices.update([message_index - 1, message_index, message_index + 1])
    chunk_indices -= {-1, 0}
    final_indices = sorted(chunk_indices)

    if secondary_database:
        query = (
            select(Message.external_id, Message.sender_name)
            .where(Message.message_index.in_(final_indices), Message.chat_id == chat_id)
            .order_by(Message.message_index)
        )
        query_results = session.execute(query).all()
        external_ids = [row[0] for row in query_results]
        sender_names = [row[1] for row in query_results]
        db_messages = get_messages_from_external_db(
            external_ids,
            secondary_database_url,
            secondary_database_encryption_method,
            secondary_database_encryption_key,
        )
        if include_names:
            messages = []
            for name, message in zip(sender_names, db_messages, strict=True):
                if name and message:
                    messages.append(f"{name}: {message}")
                elif message:
                    messages.append(message)
        else:
            messages = [message for message in db_messages if message is not None]

    else:
        query = (
            select(Message.message, Message.sender_name)
            .where(Message.message_index.in_(final_indices), Message.chat_id == chat_id)
            .order_by(Message.message_index)
        )
        query_results = session.execute(query).all()
        db_messages = [row[0] for row in query_results]
        sender_names = [row[1] for row in query_results]
        if include_names:
            messages = []
            for name, message in zip(sender_names, db_messages, strict=True):
                if name:
                    messages.append(f"{name}: {message}")
                else:
                    messages.append(message)
        else:
            messages = db_messages

    if len(messages) <= 1:
        return None

    prompt = ""
    reversed_messages = []

    for message in messages[::-1]:
        reversed_messages.append(message)
        messages_text = "\n".join(reversed_messages[::-1])
        new_prompt = summarization_settings["prompt"].format(
            term=knowledge_entry.entity,
            previous_summary=summary,
            messages=messages_text,
            bos_token=summarization_settings["bos_token"],
            input_sequence=input_sequence,
            input_suffix=input_suffix,
            output_sequence=output_sequence,
        )
        new_tokens = count_context(new_prompt, summarization_backend, summarization_url, summarization_auth)

        if new_tokens > max_context:
            break
        else:
            prompt = new_prompt

    return prompt


@celery_app.task(base=Singleton, lock_expiry=60)
def summarize(
    term: str,
    label: str,
    chat_id: int,
    api_settings: dict,
    summarization_settings: dict,
    secondary_database_settings: dict,
    db_engine: str,
    include_names: bool = True,
    max_retries: int = 50,
    retry_interval: int = 1,
) -> None:
    db = create_engine(db_engine)
    summarization_url = api_settings["url"]
    summarization_backend = api_settings["backend"]
    summarization_auth = api_settings["auth_key"]
    limit_rate = summarization_settings["limit_rate"]
    context_len = api_settings["context_length"]
    response_len = summarization_settings["max_tokens"]

    with Session(db) as session:
        knowledge_entry = get_knowledge_entity(term, chat_id, session)

        if knowledge_entry.update_count < limit_rate:  # Don't summarize if it's below the limit
            general_logger.info("Skipping entry to summarize, messages amount below limit rate")
            return None

        max_prompt_context = context_len - response_len
        prompt = make_summary_prompt(
            session,
            knowledge_entry,
            max_prompt_context,
            api_settings,
            summarization_settings,
            secondary_database_settings,
            include_names,
        )
        if prompt is None:  # Edge case of having 1 message for summary, only may happen at start of chat
            general_logger.info("Skipping entry to summarize, only 1 message for term exists")
            return None
        generation_params = {
            "max_length": response_len,
            "max_tokens": response_len,
            "truncation_length": context_len,
            "max_context_length": context_len,
        }
        generation_params.update(summarization_settings["params"])
        additional_stops = [
            api_settings["input_sequence"].strip(),
            api_settings["output_sequence"].strip(),
            api_settings["first_output_sequence"].strip(),
            api_settings["last_output_sequence"].strip(),
            api_settings["input_suffix"].strip(),
            api_settings["output_suffix"].strip(),
        ]
        additional_stops = [stop for stop in additional_stops if stop]
        generation_params["stop"].extend(additional_stops)
        generation_params["stop_sequence"].extend(additional_stops)
        summary_text, request_json = generate_text(
            prompt,
            generation_params,
            summarization_backend,
            summarization_url,
            summarization_auth,
            max_retries,
            retry_interval,
        )
        summary_text = summary_text.replace("\n\n", "\n")
        summary_text = f"[ {knowledge_entry.entity}: {summary_text} ]"
        knowledge_entry.summary = summary_text
        knowledge_entry.token_count = count_context(
            summary_text, summarization_backend, summarization_url, summarization_auth
        )
        knowledge_entry.update_count = 1
        summary_logger.debug(f"({knowledge_entry.token_count} tokens){term} ({label}): {summary_text}\n{request_json}")
        session.commit()
