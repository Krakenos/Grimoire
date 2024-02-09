import re
import timeit

import spacy
from spacy.tokens import DocBin
from sqlalchemy import desc, create_engine
from sqlalchemy.orm import Session

from memoir.common.llm_helpers import count_context
from memoir.common.loggers import general_logger, context_logger
from memoir.common.utils import orm_get_or_create
from memoir.core.settings import MODEL_INPUT_SEQUENCE, MODEL_OUTPUT_SEQUENCE, DB_ENGINE, CONTEXT_PERCENTAGE, \
    MAIN_API_BACKEND, MAIN_API_URL, MAIN_API_AUTH
from memoir.core.tasks import summarize
from memoir.db.models import Message, Knowledge

nlp = spacy.load("en_core_web_trf")
db = create_engine(DB_ENGINE)


def save_messages(messages: list, docs: list, chat_id: str, session) -> list:
    """
    Saves messages to and returns indices of new messages
    :param messages:
    :param docs:
    :param chat_id:
    :param session:
    :return: indices of new messages
    """
    new_messages_indices = []
    for message_index, message in enumerate(messages):
        message_exists = session.query(Message.id).filter_by(message=message, chat_id=chat_id).first() is not None
        if not message_exists:
            chat_exists = session.query(Message.id).filter_by(chat_id=chat_id).first() is not None
            if chat_exists:
                latest_index = session.query(Message.message_index).filter_by(chat_id=chat_id).order_by(
                    desc(Message.message_index)).first()[0]
                current_index = latest_index + 1
            else:
                current_index = 1
            spacy_doc = docs[message_index]
            doc_bin = DocBin()
            doc_bin.add(spacy_doc)
            bytes_data = doc_bin.to_bytes()
            new_message = Message(message=message, chat_id=chat_id, message_index=current_index, spacy_doc=bytes_data)
            new_messages_indices.append(message_index)
            session.add(new_message)
            session.commit()
    return new_messages_indices


def add_missing_docs(messages, docs, session):
    for index, message in messages:
        spacy_doc = docs[index]
        doc_bin = DocBin()
        doc_bin.add(spacy_doc)
        bytes_data = doc_bin.to_bytes()
        message.spacy_doc = bytes_data
        session.add(message)
    session.commit()


def get_docs(messages, chat_id, session):
    docs = []
    processing_indices = []
    to_process = []
    messages_to_update = []
    for index, message in enumerate(messages):
        db_message = session.query(Message).filter_by(chat_id=chat_id, message=message).first()
        if db_message is not None:
            if db_message.spacy_doc is not None:
                doc_bin = DocBin().from_bytes(db_message.spacy_doc)
                spacy_doc = list(doc_bin.get_docs(nlp.vocab))[0]
                docs.append((index, spacy_doc))
            else:  # if something went wrong and message doesn't have doc
                messages_to_update.append((index, db_message))
                processing_indices.append(index)
                to_process.append(message)
        else:
            processing_indices.append(index)
            to_process.append(message)
    new_docs = list(nlp.pipe(to_process))
    new_docs = [(message_index, new_docs[index]) for index, message_index in enumerate(processing_indices)]
    docs.extend(new_docs)
    docs = sorted(docs, key=lambda x: x[0])
    docs = [doc[1] for doc in docs]  # removing indices
    add_missing_docs(messages_to_update, docs, session)
    return docs


def process_prompt(prompt, chat, context_length):
    start_time = timeit.default_timer()
    banned_labels = ['DATE', 'CARDINAL', 'ORDINAL', 'TIME']
    pattern = re.escape(MODEL_INPUT_SEQUENCE) + r'|' + re.escape(MODEL_OUTPUT_SEQUENCE)
    messages = re.split(pattern, prompt)
    messages = [message.strip() for message in messages]  # remove trailing newlines
    with Session(db) as session:
        doc_time = timeit.default_timer()
        docs = get_docs(messages, chat, session)
        doc_end_time = timeit.default_timer()
        general_logger.debug(f'Creating spacy docs {doc_end_time - doc_time} seconds')
        last_messages = messages[1:-2]
        last_docs = docs[1:-2]
        new_message_indices = save_messages(last_messages, last_docs, chat, session)
        get_named_entities(chat, last_docs, session)
    docs_to_summarize = [last_docs[index] for index in new_message_indices]
    for doc in docs_to_summarize:
        for entity in set(doc.ents):
            if entity.label_ not in banned_labels:
                general_logger.debug(f'{entity.text}, {entity.label_}, {spacy.explain(entity.label_)}')
                summarize.delay(entity.text.lower(), entity.label_, chat)
    new_prompt = fill_context(prompt, chat, docs, context_length)
    end_time = timeit.default_timer()
    general_logger.info(f'Prompt processing time: {end_time - start_time}s')
    context_logger.debug(f'Old prompt: \n{prompt}\n\nNew prompt: \n{new_prompt}')
    return new_prompt


def get_named_entities(chat, docs, session):
    banned_labels = ['DATE', 'CARDINAL', 'ORDINAL', 'TIME']
    for doc in docs:
        db_message = orm_get_or_create(session, Message, message=doc.text)
        ent_list = [(str(ent), ent.label_) for ent in doc.ents if ent.label_ not in banned_labels]
        unique_ents = list(set(ent_list))
        for ent, ent_label in unique_ents:
            knowledge_entity = session.query(Knowledge).filter(Knowledge.entity.ilike(ent),
                                                               Knowledge.entity_type == 'NAMED ENTITY',
                                                               Knowledge.chat_id == chat).first()
            if not knowledge_entity:
                knowledge_entity = Knowledge(entity=ent,
                                             entity_label=ent_label,
                                             entity_type='NAMED ENTITY',
                                             chat_id=chat)
                session.add(knowledge_entity)
                session.commit()
            knowledge_entity.messages.append(db_message)
            session.add(knowledge_entity)
            session.commit()


def fill_context(prompt, chat, docs, context_size):
    max_context = context_size
    max_memoir_context = max_context * CONTEXT_PERCENTAGE
    banned_labels = ['DATE', 'CARDINAL', 'ORDINAL', 'TIME']
    pattern = re.escape(MODEL_INPUT_SEQUENCE) + r'|' + re.escape(MODEL_OUTPUT_SEQUENCE)
    pattern_with_delimiters = f'({pattern})'
    messages = re.split(pattern, prompt)
    messages_with_delimiters = re.split(pattern_with_delimiters, prompt)
    prompt_definitions = messages[0]  # first portion should always be instruction and char definitions
    full_ent_list = []
    for doc in docs:
        general_logger.debug(doc.text_with_ws)
        ent_list = [(str(ent), ent.label_) for ent in doc.ents if ent.label_ not in banned_labels]
        full_ent_list += ent_list
    unique_ents = list(dict.fromkeys(full_ent_list[::-1]))  # unique ents ordered from bottom of context
    summaries = []
    with Session(db) as session:
        for ent in unique_ents:
            query = session.query(Knowledge).filter(Knowledge.entity.ilike(ent[0]),
                                                    Knowledge.entity_type == 'NAMED ENTITY',
                                                    Knowledge.chat_id == chat,
                                                    Knowledge.entity_label == ent[1], Knowledge.summary.isnot(None),
                                                    Knowledge.summary.isnot(''), Knowledge.token_count.isnot(None),
                                                    Knowledge.token_count.isnot(0))
            instance = query.scalar()
            if instance is not None:
                summaries.append((instance.summary, instance.token_count, instance.entity))
    memoir_estimated_tokens = sum([summary_tuple[1] for summary_tuple in summaries])
    while memoir_estimated_tokens > max_memoir_context:
        summaries.pop()
        memoir_estimated_tokens = sum([summary_tuple[1] for summary_tuple in summaries])
    memoir_text = ''
    for summary in summaries:
        memoir_text = memoir_text + f'[ {summary[2]}: {summary[0]} ]\n'
    memoir_text_len = count_context(text=memoir_text, api_type=MAIN_API_BACKEND, api_url=MAIN_API_URL,
                                    api_auth=MAIN_API_AUTH)
    definitions_context_len = count_context(text=prompt_definitions, api_type=MAIN_API_BACKEND, api_url=MAIN_API_URL,
                                            api_auth=MAIN_API_AUTH)
    max_chat_context = max_context - definitions_context_len - memoir_text_len
    starting_message = 1
    messages_text = ''.join(messages_with_delimiters[starting_message:])
    messages_len = count_context(text=messages_text, api_type=MAIN_API_BACKEND, api_url=MAIN_API_URL,
                                 api_auth=MAIN_API_AUTH)
    while messages_len > max_chat_context:
        starting_message += 2
        messages_text = ''.join(messages_with_delimiters[starting_message:])
        messages_len = count_context(text=messages_text, api_type=MAIN_API_BACKEND, api_url=MAIN_API_URL,
                                     api_auth=MAIN_API_AUTH)
    final_prompt = ''.join([prompt_definitions, memoir_text, messages_text])
    return final_prompt
