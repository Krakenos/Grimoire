import timeit
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain

import numpy as np
import spacy
from rapidfuzz import fuzz
from rapidfuzz import process as fuzz_process
from rapidfuzz import utils as fuzz_utils
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload, with_loader_criteria

from grimoire.api.schemas.grimoire import KnowledgeData
from grimoire.common.loggers import general_logger
from grimoire.common.utils import time_execution
from grimoire.core.settings import settings
from grimoire.core.tasks import summarize
from grimoire.db.models import Chat, Knowledge, Message, SpacyNamedEntity, User
from grimoire.db.queries import get_knowledge_entities


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
    messages: list[str],
    messages_external_ids: list[str],
    entity_dict: dict[str, list[NamedEntity]],
    chat: Chat,
    session: Session,
) -> tuple[list[int], Chat]:
    """
    Saves messages to and returns indices of new messages
    :param messages:
    :param messages_external_ids:
    :param entity_dict:
    :param chat:
    :param session:
    :return: indices of new messages and updated chat object
    """
    new_messages_indices = []
    if settings["secondary_database"]["enabled"]:
        db_external_ids = [message.external_id for message in chat.messages]
        message_number = 0
        for index, external_id in enumerate(messages_external_ids):
            if external_id not in db_external_ids:
                new_messages_indices.append(index)

        if chat.messages:
            query = select(func.max(Message.message_index)).where(Message.chat_id == chat.id)
            message_number = session.execute(query).one()[0]

        message_number += 1

        for index in new_messages_indices:
            message_to_add = messages[index]
            named_entities = entity_dict[message_to_add]
            db_named_entities = [
                SpacyNamedEntity(entity_name=named_ent.name, entity_label=named_ent.label)
                for named_ent in named_entities
            ]
            chat.messages.append(
                Message(
                    external_id=messages_external_ids[index],
                    message_index=message_number,
                    spacy_named_entities=db_named_entities,
                )
            )
            message_number += 1

        session.add(chat)
        session.commit()
        session.refresh(chat)

        return new_messages_indices, chat

    else:
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
                SpacyNamedEntity(entity_name=named_ent.name, entity_label=named_ent.label)
                for named_ent in named_entities
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
def get_named_entities(
    messages: list[str], messages_external_ids: list[str], chat: Chat
) -> tuple[list[list[NamedEntity]], dict[str, list[NamedEntity]]]:
    entity_list = []
    entity_dict = {}
    to_process = []
    banned_labels = ["DATE", "CARDINAL", "ORDINAL", "TIME", "QUANTITY"]
    external_id_map = {}

    if settings["secondary_database"]["enabled"]:
        for message, external_id in zip(messages, messages_external_ids, strict=True):
            external_id_map[external_id] = message

    for message in chat.messages:
        named_entities = []
        if message.spacy_named_entities:
            named_entities = [
                NamedEntity(name=ent.entity_name, label=ent.entity_label) for ent in message.spacy_named_entities
            ]

        entity_dict[message.message] = named_entities

        if message.external_id and external_id_map:
            entity_dict[external_id_map[message.external_id]] = named_entities
        elif message.message:
            entity_dict[message.message] = named_entities
        else:
            raise ValueError("No external id or message provided")

    for message in messages:
        if message not in entity_dict:
            to_process.append(message)

    new_docs = list(nlp.pipe(to_process))

    for text, doc in zip(to_process, new_docs, strict=True):
        entities = [NamedEntity(ent.text, ent.label_) for ent in doc.ents if ent.label_ not in banned_labels]
        entity_dict[text] = entities
    for message in messages:
        entity_list.append(entity_dict[message])

    return entity_list, entity_dict


@time_execution
def get_user(user_id: str | None, session: Session) -> User:
    if user_id:
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
def get_chat(
    user: User, chat_id: str, current_messages: list[str], messages_external_ids: list[str], session: Session
) -> Chat:
    query = (
        select(Chat)
        .where(Chat.external_id == chat_id, Chat.user_id == user.id)
        .options(selectinload(Chat.messages), with_loader_criteria(Message, Message.message.in_(current_messages)))
    )

    if settings["secondary_database"]["enabled"]:
        query = (
            select(Chat)
            .where(Chat.external_id == chat_id, Chat.user_id == user.id)
            .options(
                selectinload(Chat.messages),
                with_loader_criteria(Message, Message.external_id.in_(messages_external_ids)),
            )
        )

    chat = session.scalars(query).first()
    if chat is None:
        chat = Chat(external_id=chat_id, user_id=user.id)
        session.add(chat)
        session.commit()

    return chat


def filter_similar_entities(entity_names: list[str]) -> dict[str, str]:
    result_matrix = fuzz_process.cdist(
        entity_names,
        entity_names,
        scorer=fuzz.WRatio,
        processor=fuzz_utils.default_process,
        score_cutoff=settings["match_distance"],
    )
    found_score_cords = np.argwhere(result_matrix)
    relation_dict = defaultdict(lambda: [])
    results = {}

    for cords in found_score_cords:
        x = cords[0]
        y = cords[1]
        relation_dict[entity_names[x]].append((entity_names[y], result_matrix[x][y]))

    entity_groups = set()
    entity_mean_score = {}

    for entity, related_entities in relation_dict.items():
        if len(related_entities) > 1:
            entity_mean_score[entity] = np.mean([ent[1] for ent in related_entities])
            related_names = [ent[0] for ent in related_entities]
            entity_groups.add(frozenset([entity, *related_names]))
        else:
            results[entity] = related_entities[0][0]

    for entity_group in entity_groups:
        mean_scores = [(entity, entity_mean_score[entity]) for entity in entity_group]
        # sorts by highest score, then longest entity, then lexically
        sorted_entities = sorted(mean_scores, key=lambda x: (-x[1], -len(x[0]), x[0]))
        top_name, _ = sorted_entities[0]

        for entity in entity_group:
            results[entity] = top_name
    return results


@time_execution
def save_named_entities(
    chat: Chat,
    entity_list: list[list[NamedEntity]],
    entity_dict: dict[str, list[NamedEntity]],
    external_id_map: dict[str, str],
    session: Session,
) -> None:
    unique_ents: list[NamedEntity] = list(set(chain(*entity_list)))
    unique_ent_names = list({ent.name for ent in unique_ents})
    ent_labels = {ent.name: ent.label for ent in unique_ents}
    similarity_dict = filter_similar_entities(unique_ent_names)
    filtered_ent_names = list(similarity_dict.values())
    knowledge_entries = get_knowledge_entities(filtered_ent_names, chat.id, session)
    found_knowledge_entries = [entry for entry in knowledge_entries if entry is not None]
    db_entry_names = [entry.entity if entry is not None else None for entry in knowledge_entries]
    new_knowledge = []

    for ent_name, db_object in zip(filtered_ent_names, db_entry_names, strict=True):
        if db_object is None:
            knowledge_entity = Knowledge(
                entity=ent_name,
                entity_label=ent_labels[ent_name],
                entity_type="NAMED ENTITY",
                chat_id=chat.id,
                update_count=0,
            )
            new_knowledge.append(knowledge_entity)

    session.add_all(new_knowledge)
    session.commit()

    knowledge_dict = {knowledge.entity: knowledge for knowledge in [*found_knowledge_entries, *new_knowledge]}

    # Link new messages to knowledge and update counter
    for db_message in chat.messages:
        if settings["secondary_database"]["enabled"]:
            message_identifier = external_id_map[db_message.external_id]
        else:
            message_identifier = db_message.message

        message_ents = entity_dict[message_identifier]
        ent_names: list[str] = list({ent.name for ent in message_ents})

        for ent in ent_names:
            coresponding_entity = similarity_dict[ent]
            if db_message not in knowledge_dict[coresponding_entity].messages:
                knowledge_dict[coresponding_entity].messages.append(db_message)
                knowledge_dict[coresponding_entity].update_count += 1

    session.add_all(knowledge_dict.values())
    session.commit()


def generate_grimoire_entries(max_grimoire_context: int, summaries: list[tuple[str, int, str]]) -> list[str]:
    grimoire_estimated_tokens = sum([summary_tuple[1] for summary_tuple in summaries])

    while grimoire_estimated_tokens > max_grimoire_context:
        summaries.pop()
        grimoire_estimated_tokens = sum([summary_tuple[1] for summary_tuple in summaries])

    grimoire_entries = [f"{summary[0]}\n" for summary in summaries]
    return grimoire_entries


def get_summaries(chat: Chat, unique_ents: list[tuple[str, str]], session: Session) -> list[tuple[str, int, str]]:
    ent_names = [name for name, _ in unique_ents]
    knowledge_ents = get_knowledge_entities(ent_names, chat.id, session)
    summaries = [
        (ent.summary, ent.token_count, ent.entity) for ent in knowledge_ents if ent is not None and ent.summary
    ]
    return summaries


def get_ordered_entities(entity_list: list[list[NamedEntity]]) -> list[tuple[str, str]]:
    full_ent_list = []

    for entities in entity_list:
        ent_list = [(ent.name, ent.label) for ent in entities]
        full_ent_list += ent_list

    unique_ents = list(dict.fromkeys(full_ent_list[::-1]))  # unique ents ordered from bottom of context
    return unique_ents


def process_request(
    external_chat_id: str,
    chat_texts: list[str],
    messages_external_ids: list[str],
    db_session,
    external_user_id: str | None = None,
    token_limit: int | None = None,
):
    start_time = timeit.default_timer()
    excluded_messages = 4

    user = get_user(external_user_id, db_session)
    chat = get_chat(user, external_chat_id, chat_texts, messages_external_ids, db_session)

    external_message_map = {}

    if settings["secondary_database"]["enabled"]:
        external_message_map = dict(zip(messages_external_ids, chat_texts, strict=True))

    doc_time = timeit.default_timer()
    entity_list, entity_dict = get_named_entities(chat_texts, messages_external_ids, chat)
    doc_end_time = timeit.default_timer()
    general_logger.debug(f"Getting named entities {doc_end_time - doc_time} seconds")
    last_messages = chat_texts[:-excluded_messages]  # exclude last few messages from saving
    last_external_ids = messages_external_ids[:-excluded_messages]
    last_entities = entity_list[:-excluded_messages]
    new_message_indices, chat = save_messages(last_messages, last_external_ids, entity_dict, chat, db_session)
    save_named_entities(chat, entity_list, entity_dict, external_message_map, db_session)

    entities_to_summarize = [last_entities[index] for index in new_message_indices]

    for entities in entities_to_summarize:
        for entity in set(entities):
            general_logger.debug(f"{entity.name}, {entity.label}, {spacy.explain(entity.label)}")
            summarize.delay(
                entity.name.lower(),
                entity.label,
                chat.id,
                settings["side_api"],
                settings["summarization"],
                settings["secondary_database"],
                settings["DB_ENGINE"],
            )

    unique_ents = get_ordered_entities(entity_list)
    summaries = get_summaries(chat, unique_ents, db_session)

    knowledge_data = []
    if token_limit:
        current_tokens = 0
        for index, summary in enumerate(summaries, 1):
            current_tokens += summary[1]
            if current_tokens < token_limit:
                knowledge_data.append(KnowledgeData(text=summary[0], relevance=index))
            else:
                break
    else:
        knowledge_data = [KnowledgeData(text=summary[0], relevance=index) for index, summary in enumerate(summaries, 1)]

    end_time = timeit.default_timer()
    general_logger.info(f"Request processing time: {end_time - start_time}s")
    return knowledge_data
