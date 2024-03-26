import re
import timeit

import spacy
from spacy.tokens import DocBin
from sqlalchemy import desc, create_engine, select
from sqlalchemy.orm import Session

from grimoire.api.request_models import Instruct, GenerationData
from grimoire.common.llm_helpers import count_context
from grimoire.common.loggers import general_logger, context_logger
from grimoire.common.utils import orm_get_or_create
from grimoire.core.settings import settings
from grimoire.core.tasks import summarize
from grimoire.db.models import Message, Knowledge

nlp = spacy.load("en_core_web_trf")
db = create_engine(settings['DB_ENGINE'])


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


def get_extra_info(prompt, generation_data: GenerationData):
    floating_prompts = []
    for message in generation_data.finalMesSend:
        floating_prompts.append(message)
    return floating_prompts


def process_prompt(prompt, chat, context_length, api_type=None, generation_data=None):
    start_time = timeit.default_timer()

    if api_type is None:
        api_type = settings['main_api']['backend']

    banned_labels = ['DATE', 'CARDINAL', 'ORDINAL', 'TIME']
    floating_prompts = None

    if settings['single_api_mode']:
        summarization_api = settings['main_api'].copy()
        if api_type is not None:
            summarization_api['backend'] = api_type
    else:
        summarization_api = settings['side_api'].copy()

    if generation_data:
        floating_prompts = get_extra_info(prompt, generation_data)

    pattern = instruct_regex()
    split_prompt = re.split(pattern, prompt)
    split_prompt = [message.strip() for message in split_prompt]  # remove trailing newlines
    all_messages = split_prompt[1:-1]  # includes injected entries at depth like WI and AN
    chat_messages = []

    if floating_prompts:
        for mes_index, message in enumerate(all_messages):
            if not floating_prompts[mes_index].injected:
                chat_messages.append(message)

    with Session(db) as session:
        doc_time = timeit.default_timer()
        docs = get_docs(chat_messages, chat, session)
        doc_end_time = timeit.default_timer()
        general_logger.debug(f'Creating spacy docs {doc_end_time - doc_time} seconds')
        last_messages = chat_messages[:-1]  # exclude user prompt
        last_docs = docs[:-1]
        new_message_indices = save_messages(last_messages, last_docs, chat, session)
        save_named_entities(chat, last_docs, session)

    docs_to_summarize = [last_docs[index] for index in new_message_indices]
    new_prompt = fill_context(prompt, chat, docs, context_length, api_type)

    for doc in docs_to_summarize:
        for entity in set(doc.ents):
            if entity.label_ not in banned_labels:
                general_logger.debug(f'{entity.text}, {entity.label_}, {spacy.explain(entity.label_)}')
                summarize.delay(entity.text.lower(), entity.label_, chat, summarization_api, settings['summarization'],
                                settings['DB_ENGINE'])

    end_time = timeit.default_timer()
    general_logger.info(f'Prompt processing time: {end_time - start_time}s')
    context_logger.debug(f'Old prompt: \n{prompt}\n\nNew prompt: \n{new_prompt}')
    return new_prompt


def save_named_entities(chat, docs, session):
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


# TODO this needs a heavy rewrite
def fill_context(prompt, chat, docs, context_size, api_type):
    max_context = context_size
    max_grimoire_context = max_context * settings['context_percentage']
    banned_labels = ['DATE', 'CARDINAL', 'ORDINAL', 'TIME']
    pattern = instruct_regex()
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
            query = select(Knowledge).where(Knowledge.entity.ilike(ent[0]),
                                            Knowledge.entity_type == 'NAMED ENTITY',
                                            Knowledge.chat_id == chat,
                                            Knowledge.summary.isnot(None),
                                            Knowledge.summary.isnot(''), Knowledge.token_count.isnot(None),
                                            Knowledge.token_count.isnot(0))
            instance = session.scalars(query).first()
            if instance is not None:
                summaries.append((instance.summary, instance.token_count, instance.entity))
    grimoire_estimated_tokens = sum([summary_tuple[1] for summary_tuple in summaries])
    while grimoire_estimated_tokens > max_grimoire_context:
        summaries.pop()
        grimoire_estimated_tokens = sum([summary_tuple[1] for summary_tuple in summaries])
    grimoire_text = ''
    for summary in summaries:
        grimoire_text = grimoire_text + f'[ {summary[2]}: {summary[0]} ]\n'
    grimoire_text_len = count_context(text=grimoire_text, api_type=api_type,
                                      api_url=settings['main_api']['url'],
                                      api_auth=settings['main_api']['auth_key'])
    definitions_context_len = count_context(text=prompt_definitions, api_type=api_type,
                                            api_url=settings['main_api']['url'],
                                            api_auth=settings['main_api']['auth_key'])
    max_chat_context = max_context - definitions_context_len - grimoire_text_len
    starting_message = 1
    messages_text = ''.join(messages_with_delimiters[starting_message:])
    messages_len = count_context(text=messages_text, api_type=api_type,
                                 api_url=settings['main_api']['url'],
                                 api_auth=settings['main_api']['auth_key'])
    while messages_len > max_chat_context:
        starting_message += 2
        first_instruct = messages_with_delimiters[starting_message]
        first_output_seq = settings['main_api']['first_output_sequence']
        output_seq = settings['main_api']['output_sequence']
        separator_seq = settings['main_api']['separator_sequence']
        if first_instruct == output_seq and first_output_seq:
            messages_with_delimiters[starting_message] = first_output_seq
        elif separator_seq in first_instruct:
            messages_with_delimiters[starting_message] = first_instruct.replace(separator_seq, '')
        messages_text = ''.join(messages_with_delimiters[starting_message:])
        messages_len = count_context(text=messages_text, api_type=api_type,
                                     api_url=settings['main_api']['url'],
                                     api_auth=settings['main_api']['auth_key'])
    final_prompt = ''.join([prompt_definitions, grimoire_text, messages_text])
    return final_prompt


def update_instruct(instruct_info: Instruct):
    if instruct_info.wrap:
        input_seq = f'{instruct_info.input_sequence}\n'
        output_seq = f'\n{instruct_info.output_sequence}\n'
    else:
        input_seq = instruct_info.input_sequence
        output_seq = instruct_info.output_sequence
    settings['main_api']['input_sequence'] = input_seq
    settings['main_api']['output_sequence'] = output_seq
    settings['main_api']['first_output_sequence'] = instruct_info.first_output_sequence
    settings['main_api']['last_output_sequence'] = instruct_info.last_output_sequence
    settings['main_api']['separator_sequence'] = instruct_info.separator_sequence


def instruct_regex():
    input_seq = re.escape(settings['main_api']['input_sequence'])
    output_seq = re.escape(settings['main_api']['output_sequence'])
    first_output_seq = re.escape(settings['main_api']['first_output_sequence'])
    last_output_seq = re.escape(settings['main_api']['last_output_sequence'])
    separator_seq = re.escape(settings['main_api']['separator_sequence'])
    pattern = input_seq + r'|' + output_seq
    if last_output_seq:
        pattern += f'|{last_output_seq}'
    if first_output_seq:
        pattern += f'|{first_output_seq}'
    if separator_seq:
        pattern += f'|{separator_seq}{input_seq}'
    return pattern
