import copy
import ssl
from datetime import datetime

from celery import Celery
from celery_singleton import Singleton
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from grimoire.common.llm_helpers import generate_text, token_count
from grimoire.common.loggers import general_logger, summary_logger
from grimoire.common.redis import redis_manager
from grimoire.core.settings import ApiSettings, SecondaryDatabaseSettings, SummarizationSettings, settings
from grimoire.db.models import Knowledge, Message
from grimoire.db.queries import get_knowledge_entity
from grimoire.db.secondary_database import get_messages_from_external_db

broker_url = redis_manager.celery_broker_url()

celery_app = Celery("tasks", broker=broker_url)

if redis_manager.sentinel:
    transport_options = {"master_name": redis_manager.sentinel_master}

    if redis_manager.tls:
        transport_options["sentinel_kwargs"] = {"ssl": True, "ssl_cert_reqs": ssl.CERT_NONE}

    celery_app.conf.broker_transport_options = transport_options

celery_app.conf.task_routes = {"grimoire.core.tasks.summarize": {"queue": "summarization_queue"}}


@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    sender.add_periodic_task(300.0, queue_logging.s(), name="log queue size every 5 minutes")


def make_summary_prompt(
    session: Session,
    knowledge_entry: Knowledge,
    max_context: int,
    api_settings: ApiSettings,
    summarization_settings: SummarizationSettings,
    secondary_database_settings: SecondaryDatabaseSettings,
    prefer_local_tokenizer: bool,
    tokenizer: str,
    include_names: bool = False,
) -> str | None:
    summarization_url = api_settings.url
    summarization_backend = api_settings.backend
    summarization_auth = api_settings.auth_key

    instruct_fields = {
        "bos_token": api_settings.bos_token,
        "system_sequence": api_settings.system_sequence,
        "system_suffix": api_settings.system_suffix,
        "input_sequence": api_settings.input_sequence,
        "input_suffix": api_settings.input_suffix,
        "output_sequence": api_settings.output_sequence,
        "output_suffix": api_settings.output_suffix,
        "first_output_sequence": api_settings.first_output_sequence,
        "last_output_sequence": api_settings.last_output_sequence,
    }

    secondary_database = secondary_database_settings.enabled
    secondary_database_url = secondary_database_settings.db_engine
    secondary_database_encryption_method = secondary_database_settings.message_encryption
    secondary_database_encryption_key = secondary_database_settings.encryption_key
    chat_id = knowledge_entry.chat_id

    if knowledge_entry.summary_entry:
        summary = f"{knowledge_entry.summary_entry}\n"
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
            select(Message.external_id, Message.character.name)
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
            select(Message.message, Message.character.name)
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
        reversed_messages.append(f"{message}\n")
        messages_text = "".join(reversed_messages[::-1])
        new_prompt = summarization_settings.prompt.format(
            term=knowledge_entry.entity, previous_summary=summary, messages=messages_text, **instruct_fields
        )
        prompt_without_summary = new_prompt
        if summary:
            prompt_without_summary = new_prompt.replace(summary, "")
        splitted_prompt = prompt_without_summary.split(messages_text)
        to_tokenize = [*splitted_prompt, *reversed_messages, summary]
        to_tokenize = [text for text in to_tokenize if text != "" and text is not None]
        new_tokens = token_count(
            to_tokenize, summarization_backend, summarization_url, tokenizer, prefer_local_tokenizer, summarization_auth
        )
        sum_tokens = sum(new_tokens)
        if sum_tokens > max_context:
            break
        else:
            prompt = new_prompt

    return prompt


@celery_app.task(base=Singleton, lock_expiry=60)
def summarize(
    term: str,
    chat_id: int,
    include_names: bool = False,
    max_retries: int = 50,
    retry_interval: int = 1,
) -> None:
    from grimoire.core.vector_embeddings import get_text_embeddings

    db_engine = settings.DB_ENGINE
    api_settings = settings.summarization_api
    summarization_settings = settings.summarization
    tokenization_settings = settings.tokenization
    secondary_database_settings = settings.secondary_database
    db = create_engine(db_engine)

    summarization_url = api_settings.url
    summarization_backend = api_settings.backend
    summarization_auth = api_settings.auth_key
    limit_rate = summarization_settings.limit_rate
    context_len = api_settings.context_length
    response_len = summarization_settings.max_tokens
    prefer_local_tokenizer = tokenization_settings.prefer_local_tokenizer
    tokenizer = tokenization_settings.local_tokenizer

    with Session(db) as session:
        knowledge_entry = get_knowledge_entity(term, chat_id, session)

        if knowledge_entry is None:
            general_logger.error("Knowledge entry not found for term")
            return None

        if knowledge_entry.update_count < limit_rate:  # Don't summarize if it's below the limit
            general_logger.info("Skipping entry to summarize, messages amount below limit rate")
            return None

        if knowledge_entry.frozen:  # Don't summarize if frozen
            general_logger.info("Skipping entry to summarize, frozen entity")
            return None

        max_prompt_context = context_len - response_len
        prompt = make_summary_prompt(
            session,
            knowledge_entry,
            max_prompt_context,
            api_settings,
            summarization_settings,
            secondary_database_settings,
            prefer_local_tokenizer,
            tokenizer,
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
        generation_params.update(copy.deepcopy(summarization_settings.params))
        additional_stops = [
            api_settings.input_sequence.strip(),
            api_settings.output_sequence.strip(),
            api_settings.first_output_sequence.strip(),
            api_settings.last_output_sequence.strip(),
            api_settings.input_suffix.strip(),
            api_settings.output_suffix.strip(),
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
        summary_entry_text = f"[ {knowledge_entry.entity}: {summary_text} ]"
        summary_embedding = get_text_embeddings(summary_text)[0]
        knowledge_entry.summary = summary_text
        knowledge_entry.summary_entry = summary_entry_text
        knowledge_entry.token_count = token_count(
            [summary_entry_text],
            summarization_backend,
            summarization_url,
            tokenizer,
            prefer_local_tokenizer,
            summarization_auth,
        )[0]
        knowledge_entry.update_count = 1
        knowledge_entry.updated_date = datetime.now()
        knowledge_entry.vector_embedding = summary_embedding
        summary_logger.debug(f"({knowledge_entry.token_count} tokens){term}: {summary_text}\n{request_json}")
        # summary_logger.debug(f"#### PROMPT ####\n{prompt}\n#### RESPONSE ####\n{summary_text}")
        session.commit()


@celery_app.task
def queue_logging() -> None:
    summarization_queue = "summarization_queue"
    redis_client = redis_manager.get_client()
    queue_length = redis_client.llen(summarization_queue)
    general_logger.info(f"Summarization tasks in queue: {queue_length}")
