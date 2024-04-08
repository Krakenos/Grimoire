import copy
import re
import timeit
from typing import Type

import spacy
from spacy.tokens import DocBin, Doc
from sqlalchemy import desc, create_engine, select
from sqlalchemy.orm import Session

from grimoire.api.request_models import Instruct, GenerationData
from grimoire.api.request_models import Message as RequestMessage
from grimoire.common.llm_helpers import count_context
from grimoire.common.loggers import general_logger, context_logger
from grimoire.common.utils import orm_get_or_create
from grimoire.core.settings import settings
from grimoire.core.tasks import summarize
from grimoire.db.models import Message, Knowledge, User, Chat

if settings['prefer_gpu']:
    gpu_check = spacy.prefer_gpu()
    if gpu_check:
        general_logger.info('Running spacy on GPU')
    else:
        general_logger.info('Failed to run on gpu, defaulting to CPU')
else:
    general_logger.info('Running spacy on CPU')

nlp = spacy.load("en_core_web_trf")
db = create_engine(settings['DB_ENGINE'])


def save_messages(messages: list, docs: list, chat: Chat, session: Session) -> list[int]:
    """
    Saves messages to and returns indices of new messages
    :param messages:
    :param docs:
    :param chat:
    :param session:
    :return: indices of new messages
    """
    new_messages_indices = []
    for message_index, message in enumerate(messages):
        message_exists = session.query(Message.id).filter_by(message=message, chat_id=chat.id).first() is not None
        if not message_exists:
            chat_exists = session.query(Message.id).filter_by(chat_id=chat.id).first() is not None
            if chat_exists:
                latest_index = session.query(Message.message_index).filter_by(chat_id=chat.id).order_by(
                    desc(Message.message_index)).first()[0]
                current_index = latest_index + 1
            else:
                current_index = 1
            spacy_doc = docs[message_index]
            doc_bin = DocBin()
            doc_bin.add(spacy_doc)
            bytes_data = doc_bin.to_bytes()
            new_message = Message(message=message, chat_id=chat.id, message_index=current_index, spacy_doc=bytes_data)
            new_messages_indices.append(message_index)
            session.add(new_message)
            session.commit()
    return new_messages_indices


def add_missing_docs(messages: list[tuple[int, Type[Message]]], docs: list[Doc], session: Session) -> None:
    for index, message in messages:
        spacy_doc = docs[index]
        doc_bin = DocBin()
        doc_bin.add(spacy_doc)
        bytes_data = doc_bin.to_bytes()
        message.spacy_doc = bytes_data
        session.add(message)
    session.commit()


def get_docs(messages: list[str], chat: Chat, session: Session) -> list[Doc]:
    docs = []
    processing_indices = []
    to_process = []
    messages_to_update = []
    for index, message in enumerate(messages):
        db_message = session.query(Message).filter_by(chat_id=chat.id, message=message).first()
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


def get_extra_info(generation_data: GenerationData) -> list[RequestMessage]:
    floating_prompts = []
    for message in generation_data.finalMesSend:
        floating_prompts.append(message)
    return floating_prompts


def get_user(user_id: str | None, current_settings: dict) -> User:
    with Session(db) as session:
        if user_id and current_settings['multi_user_mode']:
            query = select(User).where(User.external_id == user_id)
        else:
            query = select(User).where(User.external_id == 'DEFAULT_USER', User.id == 1)
        result = session.scalars(query).one()
    return result


def get_chat(user: User, chat_id: str) -> Chat:
    with Session(db) as session:
        query = select(Chat).where(Chat.external_id == chat_id,
                                   Chat.user_id == user.id)
        chat = session.scalars(query).first()
        if chat is None:
            chat = Chat(external_id=chat_id, user_id=user.id)
            session.add(chat)
            session.commit()

    return chat


def process_prompt(prompt: str,
                   chat_id: str,
                   context_length: int,
                   api_type: str | None = None,
                   generation_data: GenerationData | None = None,
                   user_id: str | None = None,
                   current_settings: dict | None = None) -> str:
    start_time = timeit.default_timer()

    if current_settings is None:
        current_settings = copy.deepcopy(settings)

    if api_type is None:
        api_type = current_settings['main_api']['backend']



    banned_labels = ['DATE', 'CARDINAL', 'ORDINAL', 'TIME']
    floating_prompts = None

    if current_settings['single_api_mode']:
        summarization_api = current_settings['main_api'].copy()
        if api_type is not None:
            summarization_api['backend'] = api_type
    else:
        summarization_api = current_settings['side_api'].copy()

    if generation_data:
        floating_prompts = get_extra_info(generation_data)

    pattern = instruct_regex(current_settings)
    split_prompt = re.split(pattern, prompt)
    split_prompt = [message.strip() for message in split_prompt]  # remove trailing newlines
    all_messages = split_prompt[1:-1]  # includes injected entries at depth like WI and AN
    chat_messages = []

    if floating_prompts:
        for mes_index, message in enumerate(all_messages):
            if not floating_prompts[mes_index].injected:
                chat_messages.append(message)
    else:
        chat_messages = all_messages

    user = get_user(user_id, current_settings)
    chat = get_chat(user, chat_id)

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
    new_prompt = fill_context(prompt, floating_prompts, chat, docs, context_length, api_type, current_settings)

    for doc in docs_to_summarize:
        for entity in set(doc.ents):
            if entity.label_ not in banned_labels:
                general_logger.debug(f'{entity.text}, {entity.label_}, {spacy.explain(entity.label_)}')
                summarize.delay(entity.text.lower(),
                                entity.label_,
                                chat.id,
                                summarization_api,
                                current_settings['summarization'],
                                current_settings['DB_ENGINE'])

    end_time = timeit.default_timer()
    general_logger.info(f'Prompt processing time: {end_time - start_time}s')
    context_logger.debug(f'Old prompt: \n{prompt}\n\nNew prompt: \n{new_prompt}')
    return new_prompt


def save_named_entities(chat: Chat, docs: list[Doc], session: Session) -> None:
    banned_labels = ['DATE', 'CARDINAL', 'ORDINAL', 'TIME']
    for doc in docs:
        db_message = orm_get_or_create(session, Message, message=doc.text)
        ent_list = [(str(ent), ent.label_) for ent in doc.ents if ent.label_ not in banned_labels]
        unique_ents = list(set(ent_list))
        for ent, ent_label in unique_ents:
            knowledge_entity = session.query(Knowledge).filter(Knowledge.entity.ilike(ent),
                                                               Knowledge.entity_type == 'NAMED ENTITY',
                                                               Knowledge.chat_id == chat.id).first()
            if not knowledge_entity:
                knowledge_entity = Knowledge(entity=ent,
                                             entity_label=ent_label,
                                             entity_type='NAMED ENTITY',
                                             chat_id=chat.id)
                session.add(knowledge_entity)
                session.commit()
            knowledge_entity.messages.append(db_message)
            knowledge_entity.update_count += 1
            session.add(knowledge_entity)
            session.commit()


# TODO Check if tokenization can be done better, waiting for the request is majority of time that takes it to process
#   potentially prefer using local tokenizer
def fill_context(prompt: str,
                 floating_prompts: list[RequestMessage],
                 chat: Chat,
                 docs: list[Doc],
                 context_size: int,
                 api_type: str,
                 current_settings: dict) -> str:
    max_context = context_size
    max_grimoire_context = max_context * current_settings['context_percentage']
    banned_labels = ['DATE', 'CARDINAL', 'ORDINAL', 'TIME']
    pattern = instruct_regex(current_settings)
    pattern_with_delimiters = f'({pattern})'
    messages = re.split(pattern, prompt)
    messages_with_delimiters = re.split(pattern_with_delimiters, prompt)
    prompt_definitions = messages[0]  # first portion should always be instruction and char definitions
    injected_prompt_indices = get_injected_indices(floating_prompts)
    unique_ents = get_ordered_entities(banned_labels, docs)
    summaries = get_summaries(chat, unique_ents)

    grimoire_text, grimoire_text_len = generate_grimoire_text(api_type, max_grimoire_context,
                                                              summaries, current_settings)

    definitions_context_len = count_context(text=prompt_definitions, api_type=api_type,
                                            api_url=current_settings['main_api']['url'],
                                            api_auth=current_settings['main_api']['auth_key'])

    max_chat_context = max_context - definitions_context_len - grimoire_text_len

    context_overflow, messages_text, min_message_context = chat_messages_culling(api_type, injected_prompt_indices,
                                                                                 max_chat_context,
                                                                                 messages_with_delimiters,
                                                                                 current_settings)

    # Grimoire + WI context overflow
    if context_overflow:
        general_logger.warning(f'Context overflow after culling messages. Trimming grimoire entries')
        grimoire_text = grimoire_entries_culling(api_type, definitions_context_len, grimoire_text, grimoire_text_len,
                                                 max_context, min_message_context, current_settings)

    if grimoire_text is None:
        general_logger.warning(f'No Grimoire entries. Passing the original prompt.')
        return prompt

    final_prompt = ''.join([prompt_definitions, grimoire_text, messages_text])
    return final_prompt


def grimoire_entries_culling(api_type: str,
                             definitions_context_len: int,
                             grimoire_text: str,
                             grimoire_text_len: int,
                             max_context: int,
                             min_message_context: int,
                             current_settings: dict) -> str | None:
    starting_grimoire_index = 0
    max_grimoire_context = max_context - definitions_context_len - min_message_context
    grimoire_entry_list = grimoire_text.splitlines()
    max_grimoire_index = len(grimoire_entry_list) - 1

    # This should never happen unless there is some bug in frontend, and it sent prompt that's above context window
    if max_grimoire_context < 0:
        general_logger.error(f'Prompt is above declared max context.')
        return None

    while max_grimoire_context < grimoire_text_len and starting_grimoire_index < max_grimoire_index:
        starting_grimoire_index += 1
        grimoire_text = '\n'.join(grimoire_entry_list[starting_grimoire_index:])
        grimoire_text_len = count_context(text=grimoire_text, api_type=api_type,
                                          api_url=current_settings['main_api']['url'],
                                          api_auth=current_settings['main_api']['auth_key'])

    return grimoire_text


def chat_messages_culling(api_type: str,
                          injected_prompt_indices: list[int],
                          max_chat_context: int,
                          messages_with_delimiters: list[str],
                          current_settings: dict) -> tuple[bool, str, int]:
    first_instruct_index = 1
    messages_text = ''.join(messages_with_delimiters[first_instruct_index:])
    messages_len = count_context(text=messages_text, api_type=api_type,
                                 api_url=current_settings['main_api']['url'],
                                 api_auth=current_settings['main_api']['auth_key'])
    messages_to_merge = {original_index: text for original_index, text in enumerate(messages_with_delimiters)}
    messages_to_merge.pop(0)
    context_overflow = False
    max_index = max(messages_to_merge.keys()) - current_settings['preserved_messages'] * 2
    min_message_context = 0

    while messages_len > max_chat_context:
        first_message_index = first_instruct_index + 1
        first_instruct_index += 2

        if first_message_index in injected_prompt_indices:
            continue

        if first_instruct_index > max_index:
            context_overflow = True
            min_message_context = messages_len
            break

        messages_to_merge.pop(first_instruct_index)  # pop the instruct part
        messages_to_merge.pop(first_message_index)  # pop actual message content

        messages_list = [messages_to_merge[index] for index in sorted(messages_to_merge.keys())]
        first_instruct = messages_list[0]
        first_output_seq = current_settings['main_api']['first_output_sequence']
        output_seq = current_settings['main_api']['output_sequence']
        separator_seq = current_settings['main_api']['separator_sequence']

        if first_instruct == output_seq and first_output_seq:
            messages_list[0] = first_output_seq
        elif separator_seq in first_instruct:
            messages_list[0] = first_instruct.replace(separator_seq, '')

        messages_text = ''.join(messages_list)
        messages_len = count_context(text=messages_text, api_type=api_type,
                                     api_url=current_settings['main_api']['url'],
                                     api_auth=current_settings['main_api']['auth_key'])

    return context_overflow, messages_text, min_message_context


def generate_grimoire_text(api_type: str,
                           max_grimoire_context: int,
                           summaries: list[tuple[str, int, str]],
                           current_settings: dict) -> tuple[str, int]:
    grimoire_estimated_tokens = sum([summary_tuple[1] for summary_tuple in summaries])
    grimoire_text = ''

    while grimoire_estimated_tokens > max_grimoire_context:
        summaries.pop()
        grimoire_estimated_tokens = sum([summary_tuple[1] for summary_tuple in summaries])

    for summary in summaries:
        grimoire_text = grimoire_text + f'[ {summary[2]}: {summary[0]} ]\n'

    grimoire_text_len = count_context(text=grimoire_text, api_type=api_type,
                                      api_url=current_settings['main_api']['url'],
                                      api_auth=current_settings['main_api']['auth_key'])
    return grimoire_text, grimoire_text_len


def get_summaries(chat: Chat, unique_ents: list[tuple[str, str]]) -> list[tuple[str, int, str]]:
    summaries = []
    with Session(db) as session:
        for ent in unique_ents:
            query = select(Knowledge).where(Knowledge.entity.ilike(ent[0]),
                                            Knowledge.entity_type == 'NAMED ENTITY',
                                            Knowledge.chat_id == chat.id,
                                            Knowledge.summary.isnot(None),
                                            Knowledge.token_count.isnot(None),
                                            Knowledge.token_count != 0)
            instance = session.scalars(query).first()

            if instance is not None:
                summaries.append((instance.summary, instance.token_count, instance.entity))

    return summaries


def get_ordered_entities(banned_labels: list[str], docs: list[Doc]) -> list[tuple[str, str]]:
    full_ent_list = []

    for doc in docs:
        ent_list = [(str(ent), ent.label_) for ent in doc.ents if ent.label_ not in banned_labels]
        full_ent_list += ent_list

    unique_ents = list(dict.fromkeys(full_ent_list[::-1]))  # unique ents ordered from bottom of context
    return unique_ents


def get_injected_indices(floating_prompts: list[RequestMessage]) -> list[int]:
    injected_prompt_indices = []

    for mes_index, message in enumerate(floating_prompts):
        if message.injected:
            corresponding_index = (mes_index + 1) * 2
            injected_prompt_indices.append(corresponding_index)

    return injected_prompt_indices


def update_instruct(instruct_info: Instruct) -> dict:
    new_settings = copy.deepcopy(settings)
    if instruct_info.wrap:
        input_seq = f'{instruct_info.input_sequence}\n'
        output_seq = f'\n{instruct_info.output_sequence}\n'
    else:
        input_seq = instruct_info.input_sequence
        output_seq = instruct_info.output_sequence
    new_settings['main_api']['system_sequence'] = instruct_info.system_sequence
    new_settings['main_api']['system_suffix'] = instruct_info.system_suffix
    new_settings['main_api']['input_sequence'] = input_seq
    new_settings['main_api']['input_suffix'] = instruct_info.input_suffix
    new_settings['main_api']['output_sequence'] = output_seq
    new_settings['main_api']['output_suffix'] = instruct_info.output_suffix
    new_settings['main_api']['first_output_sequence'] = instruct_info.first_output_sequence
    new_settings['main_api']['last_output_sequence'] = instruct_info.last_output_sequence
    new_settings['main_api']['collapse_newlines'] = instruct_info.collapse_newlines
    if instruct_info.collapse_newlines:
        for key, value in new_settings['main_api'].items():
            if key not in ['wrap', 'backend', 'url', 'auth'] and type(value) is str:
                new_settings['main_api'][key] = re.sub(r'\n+', '\n', value)
    return new_settings


def instruct_regex(current_settings) -> str:
    input_seq = re.escape(current_settings['main_api']['input_sequence'])
    input_suffix = re.escape(current_settings['main_api']['input_suffix'])
    output_seq = re.escape(current_settings['main_api']['output_sequence'])
    output_suffix = re.escape(current_settings['main_api']['output_suffix'])
    system_seq = re.escape(current_settings['main_api']['system_sequence'])
    system_suffix = re.escape(current_settings['main_api']['system_suffix'])
    first_output_seq = re.escape(current_settings['main_api']['first_output_sequence'])
    last_output_seq = re.escape(current_settings['main_api']['last_output_sequence'])
    pattern = input_seq + r'|' + output_seq + r'|' + system_seq
    if input_suffix:
        pattern += f'|{input_suffix}{input_seq}'
        pattern += f'|{input_suffix}{output_seq}'
        pattern += f'|{input_suffix}{system_seq}'
    if output_suffix:
        pattern += f'|{output_suffix}{input_seq}'
        pattern += f'|{output_suffix}{output_seq}'
        pattern += f'|{output_suffix}{system_seq}'
    if system_suffix:
        pattern += f'|{system_suffix}{input_seq}'
        pattern += f'|{system_suffix}{output_seq}'
        pattern += f'|{system_suffix}{system_seq}'
    if last_output_seq:
        pattern += f'|{last_output_seq}'
    if first_output_seq and first_output_seq != '\n':
        pattern += f'|{first_output_seq}'
    if current_settings['main_api']['collapse_newlines']:
        old_pattern = pattern
        pattern = re.sub(r'(\\\n)+', '\\\n', pattern)
    return pattern
