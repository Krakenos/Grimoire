import copy
import re
import timeit

import spacy
from spacy.tokens import Doc, DocBin
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload, with_loader_criteria

from grimoire.api.schemas.passthrough import GenerationData, Instruct
from grimoire.api.schemas.passthrough import Message as RequestMessage
from grimoire.common.llm_helpers import count_context, token_count
from grimoire.common.loggers import context_logger, general_logger
from grimoire.common.utils import async_time_execution, time_execution
from grimoire.core.settings import settings
from grimoire.core.tasks import summarize
from grimoire.db.models import Chat, Knowledge, Message, User

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
    messages: list[str], docs_dict: dict[str, Doc], chat: Chat, session: Session
) -> tuple[list[int], Chat]:
    """
    Saves messages to and returns indices of new messages
    :param messages:
    :param docs_dict:
    :param chat:
    :param session:
    :return: indices of new messages and updated chat object
    """
    new_messages_indices = []
    db_messages = [message.message for message in chat.messages]
    for index, message in enumerate(messages):
        if message not in db_messages:
            new_messages_indices.append(index)

    if db_messages:
        message_number = max([message.message_index for message in chat.messages])
    else:
        message_number = 0
    message_number += 1

    for index in new_messages_indices:
        message_to_add = messages[index]
        spacy_doc = docs_dict[message_to_add]
        doc_bin = DocBin()
        doc_bin.add(spacy_doc)
        bytes_data = doc_bin.to_bytes()
        chat.messages.append(Message(message=message_to_add, message_index=message_number, spacy_doc=bytes_data))
        message_number += 1

    session.add(chat)
    session.commit()

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
def get_docs(messages: list[str], chat: Chat, session: Session) -> tuple[list[Doc], dict[str, Doc]]:
    docs = []
    docs_dict = {}
    to_process = []
    messages_to_update = []
    for index, message in enumerate(chat.messages):
        if message.spacy_doc is not None:
            doc_bin = DocBin().from_bytes(message.spacy_doc)
            spacy_doc = list(doc_bin.get_docs(nlp.vocab))[0]
            docs_dict[message.message] = spacy_doc
        else:  # if something went wrong and message doesn't have doc
            messages_to_update.append(index)

    for message in messages:
        if message not in docs_dict:
            to_process.append(message)

    new_docs = list(nlp.pipe(to_process))
    for text, doc in zip(to_process, new_docs, strict=False):
        docs_dict[text] = doc
    for message in messages:
        docs.append(docs_dict[message])

    if messages_to_update:
        add_missing_docs(messages_to_update, docs_dict, chat, session)

    return docs, docs_dict


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

    banned_labels = ["DATE", "CARDINAL", "ORDINAL", "TIME"]
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
        an_message_index = len(split_prompt) - generation_data.authors_note.depth
        an_text = generation_data.authors_note.text.replace("{{char}}", generation_data.char)
        an_text = an_text.replace("{{user}}", generation_data.user)
        if an_message_index < 0:
            an_message_index = 0
        split_prompt[an_message_index] = split_prompt[an_message_index].replace(an_text, "")
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
    docs, docs_dict = get_docs(chat_messages, chat, db_session)
    doc_end_time = timeit.default_timer()
    general_logger.debug(f"Creating spacy docs {doc_end_time - doc_time} seconds")
    last_messages = chat_messages[:-1]  # exclude user prompt
    last_docs = docs[:-1]
    new_message_indices, chat = save_messages(last_messages, docs_dict, chat, db_session)
    save_named_entities(chat, last_docs, db_session)

    docs_to_summarize = [last_docs[index] for index in new_message_indices]

    new_prompt = await fill_context(
        prompt, floating_prompts, chat, db_session, docs, context_length, api_type, current_settings, generation_data
    )

    for doc in docs_to_summarize:
        for entity in set(doc.ents):
            if entity.label_ not in banned_labels:
                general_logger.debug(f"{entity.text}, {entity.label_}, {spacy.explain(entity.label_)}")
                summarize.delay(
                    entity.text.lower(),
                    entity.label_,
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
def save_named_entities(chat: Chat, docs: list[Doc], session: Session) -> None:
    banned_labels = ["DATE", "CARDINAL", "ORDINAL", "TIME"]
    message_indices = {message.message: index for index, message in enumerate(chat.messages)}
    unique_ents = []
    for doc in docs:
        ent_list = [(str(ent), ent.label_) for ent in doc.ents if ent.label_ not in banned_labels]
        unique_ents.extend(list(set(ent_list)))

    unique_ents = list(set(unique_ents))

    unique_ent_names = list({ent_name.lower() for ent_name, _ in unique_ents})
    query = select(Knowledge).where(
        func.lower(Knowledge.entity).in_(unique_ent_names),
        Knowledge.entity_type == "NAMED ENTITY",
        Knowledge.chat_id == chat.id,
    )
    knowledge_entries = session.scalars(query).all()
    db_entry_names = [entry.entity.lower() for entry in knowledge_entries]
    new_knowledge = []
    for ent, ent_label in unique_ents:
        if ent.lower() not in db_entry_names:
            knowledge_entity = Knowledge(
                entity=ent, entity_label=ent_label, entity_type="NAMED ENTITY", chat_id=chat.id, update_count=0
            )
            new_knowledge.append(knowledge_entity)
    session.add_all(new_knowledge)
    session.commit()

    knowledge_dict = {knowledge.entity: knowledge for knowledge in [*knowledge_entries, *new_knowledge]}

    for doc in docs:
        message_index = message_indices[doc.text]
        ent_list = [str(ent) for ent in doc.ents if ent.label_ not in banned_labels]
        message_ents = list(set(ent_list))

        for ent in message_ents:
            if chat.messages[message_index] not in knowledge_dict[ent].messages:
                knowledge_dict[ent].messages.append(chat.messages[message_index])
                knowledge_dict[ent].update_count += 1
    session.add_all(knowledge_dict.values())
    session.commit()


@async_time_execution
async def fill_context(
    prompt: str,
    floating_prompts: list[RequestMessage],
    chat: Chat,
    db_session: Session,
    docs: list[Doc],
    context_size: int,
    api_type: str,
    current_settings: dict,
    generation_data: GenerationData,
) -> str:
    max_context = context_size
    max_grimoire_context = max_context * current_settings["context_percentage"]
    banned_labels = ["DATE", "CARDINAL", "ORDINAL", "TIME"]
    pattern = instruct_regex(current_settings)
    pattern_with_delimiters = f"({pattern})"
    messages = re.split(pattern, prompt)
    messages_with_delimiters = re.split(pattern_with_delimiters, prompt)

    if (
        current_settings["main_api"]["system_sequence"] == ""
        and generation_data.authors_note
        and generation_data.authors_note.text
    ):
        an_message_index = len(messages) - generation_data.authors_note.depth
        an_text = generation_data.authors_note.text.replace("{{char}}", generation_data.char)
        an_text = an_text.replace("{{user}}", generation_data.user)
        if an_message_index < 0:
            an_message_index = 0
        messages[an_message_index] = messages[an_message_index].replace(an_text, "")
        messages.insert(an_message_index + 1, an_text)
        messages_with_delimiters[an_message_index * 2] = messages_with_delimiters[an_message_index * 2].replace(
            an_text, ""
        )
        messages_with_delimiters.insert(an_message_index * 2 + 1, "")
        messages_with_delimiters.insert(an_message_index * 2 + 2, an_text)

    prompt_definitions = messages[0]  # first portion should always be instruction and char definitions
    unique_ents = get_ordered_entities(banned_labels, docs)
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


# Old culling method, probably safe to remove
def grimoire_entries_culling(
    api_type: str,
    definitions_context_len: int,
    grimoire_text: str,
    grimoire_text_len: int,
    max_context: int,
    min_message_context: int,
    current_settings: dict,
) -> str | None:
    starting_grimoire_index = 0
    max_grimoire_context = max_context - definitions_context_len - min_message_context
    grimoire_entry_list = grimoire_text.splitlines()
    max_grimoire_index = len(grimoire_entry_list) - 1

    # This should never happen unless there is some bug in frontend, and it sent prompt that's above context window
    if max_grimoire_context < 0:
        general_logger.error("Prompt is above declared max context.")
        return None

    while max_grimoire_context < grimoire_text_len and starting_grimoire_index < max_grimoire_index:
        starting_grimoire_index += 1
        grimoire_text = "\n".join(grimoire_entry_list[starting_grimoire_index:])
        grimoire_text_len = count_context(
            text=grimoire_text,
            api_type=api_type,
            api_url=current_settings["main_api"]["url"],
            api_auth=current_settings["main_api"]["auth_key"],
        )

    return grimoire_text


# Old culling method, probably safe to remove
def chat_messages_culling(
    api_type: str,
    injected_prompt_indices: list[int],
    max_chat_context: int,
    messages_with_delimiters: list[str],
    current_settings: dict,
) -> tuple[bool, str, int]:
    first_instruct_index = 1
    messages_text = "".join(messages_with_delimiters[first_instruct_index:])
    messages_len = count_context(
        text=messages_text,
        api_type=api_type,
        api_url=current_settings["main_api"]["url"],
        api_auth=current_settings["main_api"]["auth_key"],
    )
    messages_to_merge = dict(enumerate(messages_with_delimiters))
    messages_to_merge.pop(0)
    context_overflow = False
    max_index = max(messages_to_merge.keys()) - current_settings["preserved_messages"] * 2
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
        first_output_seq = current_settings["main_api"]["first_output_sequence"]
        output_seq = current_settings["main_api"]["output_sequence"]
        separator_seq = current_settings["main_api"]["separator_sequence"]

        if first_instruct == output_seq and first_output_seq:
            messages_list[0] = first_output_seq
        elif separator_seq in first_instruct:
            messages_list[0] = first_instruct.replace(separator_seq, "")

        messages_text = "".join(messages_list)
        messages_len = count_context(
            text=messages_text,
            api_type=api_type,
            api_url=current_settings["main_api"]["url"],
            api_auth=current_settings["main_api"]["auth_key"],
        )

    return context_overflow, messages_text, min_message_context


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
