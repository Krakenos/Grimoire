import copy
import re
import timeit
from dataclasses import dataclass
from itertools import chain

import spacy
from spacy.tokens import Doc, DocBin
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload, with_loader_criteria

from grimoire.api.schemas.passthrough import GenerationData, Instruct
from grimoire.api.schemas.passthrough import Message as RequestMessage
from grimoire.common.llm_helpers import token_count
from grimoire.common.loggers import context_logger, general_logger
from grimoire.common.utils import async_time_execution, time_execution
from grimoire.core.settings import settings
from grimoire.core.tasks import summarize
from grimoire.db.models import Chat, Knowledge, Message, SpacyNamedEntity, User


@dataclass(frozen=True, eq=True)
class NamedEntity:
    name: str
    label: str


if settings["prefer_gpu"]:
    gpu_check = spacy.prefer_gpu()
    if gpu_check:
        general_logger.info("Running spacy on GPU")
    else:
        general_logger.info("Failed to run on gpu, defaulting to CPU")
else:
    general_logger.info("Running spacy on CPU")

nlp = spacy.load("en_core_web_trf")


@time_execution
def save_messages(
    messages: list[str], entity_dict: dict[str, list[NamedEntity]], chat: Chat, session: Session
) -> tuple[list[int], Chat]:
    """
    Saves messages to and returns indices of new messages
    :param messages:
    :param entity_dict:
    :param chat:
    :param session:
    :return: indices of new messages and updated chat object
    """
    new_messages_indices = []
    db_messages = [message.message for message in chat.messages]
    message_number = 0
    for index, message in enumerate(messages):
        if message not in db_messages:
            new_messages_indices.append(index)

    if db_messages:
        query = select(func.max(Message.message_index)).where(Message.chat_id == chat.id)
        message_number = session.execute(query).one()[0]

    message_number += 1

    for index in new_messages_indices:
        message_to_add = messages[index]
        named_entities = entity_dict[message_to_add]
        db_named_entities = [
            SpacyNamedEntity(entity_name=named_ent.name, entity_label=named_ent.label) for named_ent in named_entities
        ]
        chat.messages.append(
            Message(message=message_to_add, message_index=message_number, spacy_named_entities=db_named_entities)
        )
        message_number += 1

    session.add(chat)
    session.commit()
    session.refresh(chat)

    return new_messages_indices, chat


@time_execution
def add_missing_docs(message_indices: list[int], docs_dict: dict[str, Doc], chat: Chat, session: Session) -> None:
    for message_index in message_indices:
        doc = docs_dict[chat.messages[message_index].message]
        doc_bin = DocBin()
        doc_bin.add(doc)
        bytes_data = doc_bin.to_bytes()
        chat.messages[message_index].spacy_doc = bytes_data
    session.add(chat)
    session.commit()


@time_execution
def get_named_entities(messages: list[str], chat: Chat) -> tuple[list[list[NamedEntity]], dict[str, list[NamedEntity]]]:
    entity_list = []
    entity_dict = {}
    to_process = []
    banned_labels = ["DATE", "CARDINAL", "ORDINAL", "TIME"]
    for message in chat.messages:
        if message.spacy_named_entities:
            named_entities = [
                NamedEntity(name=ent.entity_name, label=ent.entity_label) for ent in message.spacy_named_entities
            ]
            entity_dict[message.message] = named_entities
        else:
            entity_dict[message.message] = []

    for message in messages:
        if message not in entity_dict:
            to_process.append(message)

    new_docs = list(nlp.pipe(to_process))

    for text, doc in zip(to_process, new_docs, strict=False):
        entities = [NamedEntity(ent.text, ent.label_) for ent in doc.ents if ent.label_ not in banned_labels]
        entity_dict[text] = entities
    for message in messages:
        entity_list.append(entity_dict[message])

    return entity_list, entity_dict


@time_execution
def get_extra_info(
    generation_data: GenerationData, messages: list[str], attached_an=False
) -> list[RequestMessage] | None:
    floating_prompts = []
    if generation_data.finalMesSend:
        for message in generation_data.finalMesSend:
            floating_prompts.append(message)
    elif generation_data.authors_note and generation_data.authors_note.text:
        an_index = len(messages) - generation_data.authors_note.depth
        if an_index < 0:
            an_index = 0
        author_note = generation_data.authors_note.text.replace("{{char}}", generation_data.char)
        author_note = author_note.replace("{{user}}", generation_data.user)
        for index, mes in enumerate(messages):
            if index == an_index and author_note in mes:
                floating_prompts.append(RequestMessage(message=mes, injected=True))
            else:
                floating_prompts.append(RequestMessage(message=mes, injected=False))
    else:
        return None
    return floating_prompts


@time_execution
def get_user(user_id: str | None, current_settings: dict, session: Session) -> User:
    if user_id and current_settings["multi_user_mode"]:
        query = select(User).where(User.external_id == user_id)
    else:
        query = select(User).where(User.external_id == "DEFAULT_USER", User.id == 1)
    result = session.scalars(query).first()
    if not result:
        result = User(external_id=user_id)
        session.add(result)
        session.commit()
        session.refresh(result)
    return result


@time_execution
def get_chat(user: User, chat_id: str, current_messages: list[str], session: Session) -> Chat:
    query = (
        select(Chat)
        .where(Chat.external_id == chat_id, Chat.user_id == user.id)
        .options(selectinload(Chat.messages), with_loader_criteria(Message, Message.message.in_(current_messages)))
    )
    chat = session.scalars(query).first()
    if chat is None:
        chat = Chat(external_id=chat_id, user_id=user.id)
        session.add(chat)
        session.commit()

    return chat


async def process_prompt(
    prompt: str,
    chat_id: str,
    context_length: int,
    db_session: Session,
    api_type: str | None = None,
    generation_data: GenerationData | None = None,
    user_id: str | None = None,
    current_settings: dict | None = None,
) -> str:
    start_time = timeit.default_timer()
    if current_settings is None:
        current_settings = copy.deepcopy(settings)

    if api_type is None:
        api_type = current_settings["main_api"]["backend"]

    floating_prompts = None

    if current_settings["single_api_mode"]:
        summarization_api = current_settings["main_api"].copy()
        if api_type is not None:
            summarization_api["backend"] = api_type
    else:
        summarization_api = current_settings["side_api"].copy()

    pattern = instruct_regex(current_settings)
    split_prompt = re.split(pattern, prompt)
    split_prompt = [message.strip() for message in split_prompt]  # remove trailing newlines
    chat_start_index = 1

    if split_prompt[0] == "":
        split_prompt = split_prompt[1:]

    attached_an = False

    # Split author's note if it's attached to the message
    if (
        current_settings["main_api"]["system_sequence"] == ""
        and generation_data.authors_note
        and generation_data.authors_note.text
    ):
        an_message_index = len(split_prompt) - generation_data.authors_note.depth - 1
        an_text = generation_data.authors_note.text.replace("{{char}}", generation_data.char)
        an_text = an_text.replace("{{user}}", generation_data.user)
        if an_message_index < 0:
            an_message_index = 0
        split_prompt[an_message_index] = split_prompt[an_message_index].replace(f"\n{an_text}", "")
        split_prompt.insert(an_message_index + 1, an_text)
        attached_an = True

    all_messages = split_prompt[chat_start_index:-1]  # includes injected entries at depth like WI and AN
    chat_messages = []

    if generation_data:
        floating_prompts = get_extra_info(generation_data, all_messages, attached_an)

    if floating_prompts:
        for mes_index, message in enumerate(all_messages):
            if not floating_prompts[mes_index].injected:
                chat_messages.append(message)
    else:
        chat_messages = all_messages

    user = get_user(user_id, current_settings, db_session)
    chat = get_chat(user, chat_id, chat_messages, db_session)

    doc_time = timeit.default_timer()
    entity_list, entity_dict = get_named_entities(chat_messages, chat)
    doc_end_time = timeit.default_timer()
    general_logger.debug(f"Getting named entities {doc_end_time - doc_time} seconds")
    last_messages = chat_messages[:-1]  # exclude user prompt
    last_entities = entity_list[:-1]
    new_message_indices, chat = save_messages(last_messages, entity_dict, chat, db_session)
    save_named_entities(chat, entity_list, entity_dict, db_session)

    entities_to_summarize = [last_entities[index] for index in new_message_indices]

    new_prompt = await fill_context(
        prompt,
        floating_prompts,
        chat,
        db_session,
        entity_list,
        context_length,
        api_type,
        current_settings,
        generation_data,
    )

    for entities in entities_to_summarize:
        for entity in set(entities):
            general_logger.debug(f"{entity.name}, {entity.label}, {spacy.explain(entity.label)}")
            summarize.delay(
                entity.name.lower(),
                entity.label,
                chat.id,
                summarization_api,
                current_settings["summarization"],
                current_settings["DB_ENGINE"],
            )

    end_time = timeit.default_timer()
    general_logger.info(f"Prompt processing time: {end_time - start_time}s")
    context_logger.debug(f"Old prompt: \n{prompt}\n\nNew prompt: \n{new_prompt}")
    return new_prompt


@time_execution
def save_named_entities(
    chat: Chat, entity_list: list[list[NamedEntity]], entity_dict: dict[str, list[NamedEntity]], session: Session
) -> None:
    unique_ents: list[NamedEntity] = list(set(chain(*entity_list)))

    unique_ent_names = list({ent.name.lower() for ent in unique_ents})
    query = select(Knowledge).where(
        func.lower(Knowledge.entity).in_(unique_ent_names),
        Knowledge.entity_type == "NAMED ENTITY",
        Knowledge.chat_id == chat.id,
    )
    knowledge_entries = session.scalars(query).all()
    db_entry_names = [entry.entity.lower() for entry in knowledge_entries]
    new_knowledge = []
    for ent in unique_ents:
        if ent.name.lower() not in db_entry_names:
            knowledge_entity = Knowledge(
                entity=ent.name, entity_label=ent.label, entity_type="NAMED ENTITY", chat_id=chat.id, update_count=0
            )
            new_knowledge.append(knowledge_entity)
    session.add_all(new_knowledge)
    session.commit()

    knowledge_dict = {knowledge.entity: knowledge for knowledge in [*knowledge_entries, *new_knowledge]}

    # Link new messages to knowledge and update counter
    for db_message in chat.messages:
        message_ents = entity_dict[db_message.message]
        ent_names: list[str] = list({ent.name for ent in message_ents})
        for ent in ent_names:
            if db_message not in knowledge_dict[ent].messages:
                knowledge_dict[ent].messages.append(db_message)
                knowledge_dict[ent].update_count += 1
    session.add_all(knowledge_dict.values())
    session.commit()


@async_time_execution
async def fill_context(
    prompt: str,
    floating_prompts: list[RequestMessage],
    chat: Chat,
    db_session: Session,
    entity_list: list[list[NamedEntity]],
    context_size: int,
    api_type: str,
    current_settings: dict,
    generation_data: GenerationData,
) -> str:
    max_context = context_size
    max_grimoire_context = max_context * current_settings["context_percentage"]
    pattern = instruct_regex(current_settings)
    pattern_with_delimiters = f"({pattern})"
    messages = re.split(pattern, prompt)
    messages_with_delimiters = re.split(pattern_with_delimiters, prompt)

    if (
        current_settings["main_api"]["system_sequence"] == ""
        and generation_data.authors_note
        and generation_data.authors_note.text
    ):
        an_message_index = len(messages) - generation_data.authors_note.depth - 1
        an_text = generation_data.authors_note.text.replace("{{char}}", generation_data.char)
        an_text = an_text.replace("{{user}}", generation_data.user)
        if an_message_index < 0:
            an_message_index = 0
        messages[an_message_index] = messages[an_message_index].replace(f"\n{an_text}", "")
        messages_with_delimiters[an_message_index * 2] = messages_with_delimiters[an_message_index * 2].replace(
            f"\n{an_text}", ""
        )
        messages_with_delimiters.insert(an_message_index * 2 + 1, "\n")
        messages_with_delimiters.insert(an_message_index * 2 + 2, an_text)
        messages.insert(an_message_index + 1, an_text)

    prompt_definitions = messages[0]  # first portion should always be instruction and char definitions
    unique_ents = get_ordered_entities(entity_list)
    summaries = get_summaries(chat, unique_ents, db_session)

    prompt_entries = {
        "prompt_definitions": prompt_definitions,
        "grimoire": [],
        "messages_with_delimiters": messages_with_delimiters,
        "floating_prompts": floating_prompts,
        "original_prompt": prompt,
    }

    grimoire_entries = generate_grimoire_entries(max_grimoire_context, summaries)

    prompt_entries["grimoire"] = grimoire_entries

    final_prompt = await prompt_culling(api_type, prompt_entries, context_size, current_settings)

    return final_prompt


@async_time_execution
async def prompt_culling(api_type: str, prompt_entries: dict, max_context: int, current_settings: dict) -> str:
    prompt_definitions = prompt_entries["prompt_definitions"]
    grimoire_entries = prompt_entries["grimoire"]
    messages_with_delimiters = prompt_entries["messages_with_delimiters"]
    floating_prompts = prompt_entries["floating_prompts"]

    first_output_seq = current_settings["main_api"]["first_output_sequence"]
    output_seq = current_settings["main_api"]["output_sequence"]
    system_suffix = current_settings["main_api"]["system_suffix"]
    input_suffix = current_settings["main_api"]["input_suffix"]
    output_suffix = current_settings["main_api"]["output_suffix"]

    messages_dict = dict(enumerate(messages_with_delimiters[1:]))

    injected_indices = []
    if floating_prompts:
        injected_indices = get_injected_indices(floating_prompts)

    messages_list = [messages_dict[index] for index in sorted(messages_dict.keys())]
    messages = [prompt_definitions, *grimoire_entries, *messages_list]
    token_amounts = await token_count(
        messages, api_type, current_settings["main_api"]["url"], current_settings["main_api"]["auth_key"]
    )
    current_tokens = sum(token_amounts)

    max_index = max(messages_dict.keys()) - 2 * current_settings["preserved_messages"]
    messages_start_index = 0
    starting_grimoire_index = 0
    while current_tokens > max_context:
        if messages_start_index < max_index:
            if messages_start_index not in injected_indices:
                messages_dict.pop(messages_start_index)
                messages_dict.pop(messages_start_index + 1)
                messages_list = [messages_dict[index] for index in sorted(messages_dict.keys())]
                first_instruct = min(messages_dict.keys())
                first_text = first_instruct + 1

                if first_instruct == output_seq and first_output_seq:
                    messages_list[0] = f"{first_output_seq}{first_text}"
                elif system_suffix in first_instruct:
                    messages_list[0] = messages_list[0].replace(system_suffix, "")
                elif input_suffix in first_instruct:
                    messages_list[0] = messages_list[0].replace(input_suffix, "")
                elif output_suffix in first_instruct:
                    messages_list[0] = messages_list[0].replace(output_suffix, "")

                messages = [prompt_definitions, *grimoire_entries, *messages_list]
                token_amounts = await token_count(
                    messages, api_type, current_settings["main_api"]["url"], current_settings["main_api"]["auth_key"]
                )
                current_tokens = sum(token_amounts)
            messages_start_index += 2
        elif starting_grimoire_index < grimoire_entries:
            starting_grimoire_index += 1
            messages = [prompt_definitions, *grimoire_entries[starting_grimoire_index:], *messages_list]
            current_tokens = sum(token_amounts)
        else:
            general_logger.warning("No Grimoire entries. Passing the original prompt.")
            return prompt_entries["original_prompt"]
    prompt_text = "".join(messages)
    return prompt_text


def generate_grimoire_entries(max_grimoire_context: int, summaries: list[tuple[str, int, str]]) -> list[str]:
    grimoire_estimated_tokens = sum([summary_tuple[1] for summary_tuple in summaries])

    while grimoire_estimated_tokens > max_grimoire_context:
        summaries.pop()
        grimoire_estimated_tokens = sum([summary_tuple[1] for summary_tuple in summaries])

    grimoire_entries = [f"[ {summary[2]}: {summary[0]} ]\n" for summary in summaries]
    return grimoire_entries


def get_summaries(chat: Chat, unique_ents: list[tuple[str, str]], session: Session) -> list[tuple[str, int, str]]:
    summaries = []
    lower_ent_names = [name.lower() for name, _ in unique_ents]
    query = select(Knowledge).where(
        func.lower(Knowledge.entity).in_(lower_ent_names),
        Knowledge.entity_type == "NAMED ENTITY",
        Knowledge.chat_id == chat.id,
        Knowledge.summary.isnot(None),
        Knowledge.token_count.isnot(None),
        Knowledge.token_count != 0,
    )
    knowledge_ents = session.scalars(query).all()

    for instance in knowledge_ents:
        summaries.append((instance.summary, instance.token_count, instance.entity))

    return summaries


def get_ordered_entities(entity_list: list[list[NamedEntity]]) -> list[tuple[str, str]]:
    full_ent_list = []

    for entities in entity_list:
        ent_list = [(ent.name, ent.label) for ent in entities]
        full_ent_list += ent_list

    unique_ents = list(dict.fromkeys(full_ent_list[::-1]))  # unique ents ordered from bottom of context
    return unique_ents


def get_injected_indices(floating_prompts: list[RequestMessage]) -> list[int]:
    injected_prompt_indices = []

    for mes_index, message in enumerate(floating_prompts):
        if message.injected:
            corresponding_index = mes_index * 2 + 1
            injected_prompt_indices.append(corresponding_index)

    return injected_prompt_indices


def update_instruct(instruct_info: Instruct, char_name: str | None = None, user_name: str | None = None) -> dict:
    new_settings = copy.deepcopy(settings)
    sys_suffix = instruct_info.system_suffix
    input_suffix = instruct_info.input_suffix
    output_suffix = instruct_info.output_suffix
    if instruct_info.trailing_newline:
        sys_suffix = f"{sys_suffix}\n"
        input_suffix = f"{input_suffix}\n"
        output_suffix = f"{output_suffix}\n"
    if instruct_info.wrap:
        input_seq = f"{instruct_info.input_sequence}\n"
        output_seq = f"\n{instruct_info.output_sequence}\n"
    else:
        input_seq = instruct_info.input_sequence
        output_seq = instruct_info.output_sequence

    new_settings["main_api"]["system_sequence"] = instruct_info.system_sequence
    new_settings["main_api"]["system_suffix"] = sys_suffix
    new_settings["main_api"]["input_sequence"] = input_seq
    new_settings["main_api"]["input_suffix"] = input_suffix
    new_settings["main_api"]["output_sequence"] = output_seq
    new_settings["main_api"]["output_suffix"] = output_suffix
    new_settings["main_api"]["first_output_sequence"] = instruct_info.first_output_sequence
    new_settings["main_api"]["last_output_sequence"] = instruct_info.last_output_sequence
    new_settings["main_api"]["collapse_newlines"] = instruct_info.collapse_newlines
    new_settings["main_api"]["trailing_newline"] = instruct_info.trailing_newline

    if "{{user}}" in new_settings["main_api"]["input_sequence"]:
        new_settings["main_api"]["input_sequence"] = new_settings["main_api"]["input_sequence"].replace(
            "{{user}}", user_name
        )

    if "{{char}}" in new_settings["main_api"]["output_sequence"]:
        new_settings["main_api"]["output_sequence"] = new_settings["main_api"]["output_sequence"].replace(
            "{{char}}", char_name
        )

    if instruct_info.collapse_newlines:
        for key, value in new_settings["main_api"].items():
            if key not in ["wrap", "backend", "url", "auth"] and isinstance(value, str):
                new_settings["main_api"][key] = re.sub(r"\n+", "\n", value)
    return new_settings


def instruct_regex(current_settings: dict) -> str:
    input_seq = re.escape(current_settings["main_api"]["input_sequence"])
    input_suffix = re.escape(current_settings["main_api"]["input_suffix"])
    output_seq = re.escape(current_settings["main_api"]["output_sequence"])
    output_suffix = re.escape(current_settings["main_api"]["output_suffix"])
    system_seq = re.escape(current_settings["main_api"]["system_sequence"])
    system_suffix = re.escape(current_settings["main_api"]["system_suffix"])
    first_output_seq = re.escape(current_settings["main_api"]["first_output_sequence"])
    last_output_seq = re.escape(current_settings["main_api"]["last_output_sequence"])

    patterns = [
        input_seq,
        output_seq,
        system_seq,
        f"{input_suffix}{input_seq}",
        f"{input_suffix}{output_seq}",
        f"{input_suffix}{system_seq}",
        f"{output_suffix}{input_seq}",
        f"{output_suffix}{output_seq}",
        f"{output_suffix}{system_seq}",
        f"{system_suffix}{input_seq}",
        f"{system_suffix}{output_seq}",
        f"{system_suffix}{system_seq}",
        f"{last_output_seq}",
        f"{first_output_seq}",
    ]

    unique_patterns = list(set(patterns))
    filtered_patterns = filter(lambda x: x and x not in ("\n", "\\\n", " "), unique_patterns)

    pattern = "|".join(filtered_patterns)

    if current_settings["main_api"]["collapse_newlines"]:
        pattern = re.sub(r"(\\\n)+", "\\\n", pattern)
    return pattern
