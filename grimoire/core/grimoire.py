import json
import timeit
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import chain

import numpy as np
import spacy
from rapidfuzz import fuzz
from rapidfuzz import process as fuzz_process
from rapidfuzz import utils as fuzz_utils
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload, with_loader_criteria

from grimoire.api.schemas.grimoire import ChatDataCharacter, KnowledgeData
from grimoire.common.loggers import general_logger
from grimoire.common.redis import redis_manager
from grimoire.common.utils import time_execution
from grimoire.core.settings import settings
from grimoire.core.tasks import summarize
from grimoire.core.vector_embeddings import get_text_embeddings
from grimoire.db.models import Character, CharacterTriggerText, Chat, Knowledge, Message, SpacyNamedEntity, User
from grimoire.db.queries import get_characters, get_knowledge_entities, semantic_search


@dataclass(frozen=True, eq=True)
class NamedEntity:
    name: str
    label: str


if settings.prefer_gpu:
    gpu_check = spacy.require_gpu()
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
    sender_names: list[str],
    characters_dict: dict[str, Character],
    entity_dict: dict[str, list[NamedEntity]],
    embedding_dict: dict[str, np.ndarray],
    chat: Chat,
    session: Session,
) -> tuple[list[int], Chat, list[Message]]:
    """
    Saves messages to and returns indices of new messages
    :param messages:
    :param messages_external_ids:
    :param sender_names:
    :param characters_dict:
    :param entity_dict:
    :param embedding_dict:
    :param chat:
    :param session:
    :return: indices of new messages, updated chat object, new messages objects
    """
    new_messages_indices = []
    new_messages = []
    chars = [characters_dict[name] for name in sender_names]
    if settings.secondary_database.enabled:
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
            embedding = embedding_dict[message_to_add]
            db_named_entities = [
                SpacyNamedEntity(entity_name=named_ent.name, entity_label=named_ent.label)
                for named_ent in named_entities
            ]
            new_message = Message(
                external_id=messages_external_ids[index],
                character_id=chars[index].id,
                message_index=message_number,
                vector_embedding=embedding,
                spacy_named_entities=db_named_entities,
            )
            chat.messages.append(new_message)
            new_messages.append(new_message)
            message_number += 1

        session.add(chat)
        session.commit()
        session.refresh(chat)
        return new_messages_indices, chat, new_messages

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
            embedding = embedding_dict[message_to_add]
            db_named_entities = [
                SpacyNamedEntity(entity_name=named_ent.name, entity_label=named_ent.label)
                for named_ent in named_entities
            ]
            new_message = Message(
                message=message_to_add,
                character_id=chars[index].id,
                message_index=message_number,
                vector_embedding=embedding,
                spacy_named_entities=db_named_entities,
            )
            chat.messages.append(new_message)
            new_messages.append(new_message)
            message_number += 1

        session.add(chat)
        session.commit()
        session.refresh(chat)

        return new_messages_indices, chat, new_messages


def get_cached_entities(texts: list[str]) -> list[list[NamedEntity] | None]:
    redis_client = redis_manager.get_client()
    redis_keys = [f"NAMED_ENTITIES_{text}" for text in texts]
    cached_entries = []
    for key in redis_keys:
        cached_value = redis_client.get(key)
        if cached_value is not None:
            cached_list = json.loads(cached_value)
            entity_list = [NamedEntity(**entity_dict) for entity_dict in cached_list]
            cached_entries.append(entity_list)
        else:
            cached_entries.append(None)
    return cached_entries


def cache_entities(texts: list[str], entities: list[list[NamedEntity]]) -> None:
    redis_client = redis_manager.get_client()
    redis_keys = [f"NAMED_ENTITIES_{text}" for text in texts]
    redis_values = []
    for entity_list in entities:
        redis_value = [asdict(entity) for entity in entity_list]
        redis_values.append(redis_value)

    for key, value in zip(redis_keys, redis_values, strict=True):
        redis_client.set(key, json.dumps(value), settings.redis.CACHE_EXPIRE_TIME)


@time_execution
def get_named_entities(
    messages: list[str],
    messages_external_ids: list[str],
    messages_names: list[str],
    chat: Chat,
) -> tuple[list[list[NamedEntity]], dict[str, list[NamedEntity]]]:
    entity_list = []
    entity_dict = {}
    to_check_cache = []
    to_process = []
    to_process_with_names = []
    banned_labels = ["DATE", "CARDINAL", "ORDINAL", "TIME", "QUANTITY", "PERCENT"]
    external_id_map = {}

    if settings.secondary_database.enabled:
        for message, external_id in zip(messages, messages_external_ids, strict=True):
            external_id_map[external_id] = message

    for message in chat.messages:
        named_entities = []
        if message.spacy_named_entities:
            named_entities = [
                NamedEntity(name=ent.entity_name, label=ent.entity_label) for ent in message.spacy_named_entities
            ]

        if message.external_id and external_id_map:
            entity_dict[external_id_map[message.external_id]] = named_entities
        elif message.message:
            entity_dict[message.message] = named_entities
        else:
            raise ValueError("No external id or message provided")

    for message in messages:
        if message not in entity_dict:
            to_check_cache.append(message)

    cached_values = get_cached_entities(to_check_cache)
    for message, cached in zip(to_check_cache, cached_values, strict=True):
        if cached is not None:
            entity_dict[message] = cached

    for sender_name, message in zip(messages_names, messages, strict=True):
        if message not in entity_dict:
            to_process_with_names.append(f"{sender_name}: {message}")
            to_process.append(message)

    new_docs = list(nlp.pipe(to_process_with_names))

    for text, doc in zip(to_process, new_docs, strict=True):
        entities = [NamedEntity(ent.text, ent.label_) for ent in doc.ents if ent.label_ not in banned_labels]
        entity_dict[text] = entities

    values_to_cache = [entity_dict[text] for text in to_process]
    cache_entities(to_process, values_to_cache)
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

    if settings.secondary_database.enabled:
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
        scorer=fuzz.partial_ratio,
        processor=fuzz_utils.default_process,
    )
    found_score_cords = np.argwhere(result_matrix >= settings.match_distance)
    relation_dict = defaultdict(list)
    results = {}

    score_dict = defaultdict(dict)
    for i, word_1 in enumerate(entity_names):
        for j, word_2 in enumerate(entity_names):
            score_dict[word_1][word_2] = result_matrix[i, j]

    for cords in found_score_cords:
        x = cords[0]
        y = cords[1]
        relation_dict[entity_names[x]].append((entity_names[y], result_matrix[x][y]))

    for entity, related_entities in relation_dict.items():
        if len(related_entities) > 1:
            related_names = [ent[0] for ent in related_entities]

            # Get mean score for each entity in relation to other ones in group
            mean_scores = []
            for ent_1 in related_names:
                ent_scores = [score_dict[ent_1][ent_2] for ent_2 in related_names]
                mean_scores.append((ent_1, np.mean(ent_scores)))

            # sorts by highest score, then shortest entity, then lexically
            sorted_entities = sorted(mean_scores, key=lambda x: (-x[1], len(x[0]), x[0]))
            top_name, _ = sorted_entities[0]
            results[entity] = top_name
        else:
            results[entity] = related_entities[0][0]
    return results


@time_execution
def save_named_entities(
    chat: Chat,
    entity_list: list[list[NamedEntity]],
    similarity_dict: dict[str, str],
    session: Session,
) -> tuple[dict[str, Knowledge], dict[str, str]]:
    unique_ents: list[NamedEntity] = list(set(chain(*entity_list)))
    unique_ent_names = list({ent.name for ent in unique_ents})

    ent_labels = {similarity_dict[ent.name]: ent.label for ent in unique_ents}
    filtered_ent_names = list({similarity_dict[name] for name in unique_ent_names})

    knowledge_entries = get_knowledge_entities(filtered_ent_names, chat.id, session)
    found_knowledge_entries = [entry for entry in knowledge_entries if entry is not None]
    db_entry_names = [entry.entity if entry is not None else None for entry in knowledge_entries]

    new_knowledge = []
    db_ent_map = {}

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
            db_ent_map[ent_name] = ent_name
        else:
            db_ent_map[ent_name] = db_object

    session.add_all(new_knowledge)
    session.commit()

    knowledge_dict = {knowledge.entity: knowledge for knowledge in [*found_knowledge_entries, *new_knowledge]}

    return knowledge_dict, db_ent_map


def link_knowledge(
    messages_to_update: list[Message],
    knowledge_dict: dict[str, Knowledge],
    db_entity_map: dict[str, str],
    similarity_dict: dict[str, str],
    session: Session,
) -> None:
    for message in messages_to_update:
        for entity in message.spacy_named_entities:
            corresponding_entity = similarity_dict[entity.entity_name]
            db_name = db_entity_map[corresponding_entity]
            if message not in knowledge_dict[db_name].messages:
                knowledge_dict[db_name].messages.append(message)
                knowledge_dict[db_name].update_count += 1
    session.add_all(knowledge_dict.values())
    session.commit()


def get_embeddings(
    chat: Chat, chat_texts: list[str], messages_names: list[str], external_message_map: dict
) -> dict[str, np.ndarray]:
    embedding_dict = {}

    if settings.secondary_database.enabled:
        for mes in chat.messages:
            mes_text = external_message_map[mes.external_id]
            if mes.vector_embedding is not None:
                embedding_dict[mes_text] = np.array(mes.vector_embedding)
    else:
        for mes in chat.messages:
            if mes.vector_embedding is not None:
                embedding_dict[mes.message] = np.array(mes.vector_embedding)

    to_vectorize = [
        f"{name}: {text}" for name, text in zip(messages_names, chat_texts, strict=True) if text not in embedding_dict
    ]
    new_texts = [text for text in chat_texts if text not in embedding_dict]
    new_embeddings = get_text_embeddings(to_vectorize)

    for text, embedding in zip(new_texts, new_embeddings, strict=True):
        embedding_dict[text] = embedding
    return embedding_dict


def update_characters(
    characters: list[ChatDataCharacter], chat_id: int, similarity_dict: dict[str, str], session: Session
) -> dict[str, Character]:
    char_names = [character.name for character in characters]
    db_characters = get_characters(char_names, chat_id, session)

    mapped_entities = [similarity_dict[name] if name in similarity_dict else name for name in char_names]
    trigger_strings = []
    for entity_name in mapped_entities:
        entity_trigger_strings = [
            trigger_string for trigger_string, ent in similarity_dict.items() if ent == entity_name
        ]
        if not entity_trigger_strings:
            entity_trigger_strings = [entity_name]

        trigger_strings.append(entity_trigger_strings)

    to_update = []
    character_dict = {}
    for request_char, db_char, triggers in zip(characters, db_characters, trigger_strings, strict=True):
        if db_char is None:
            new_char = Character(
                chat_id=chat_id,
                name=request_char.name,
                description=request_char.description,
                character_note=request_char.character_note,
            )

            for trigger in triggers:
                new_char.trigger_texts.append(CharacterTriggerText(text=trigger))
            to_update.append(new_char)
            character_dict[request_char.name] = new_char
        else:
            new_attributes = request_char.model_dump(exclude_unset=True, exclude_none=False)
            for key, value in new_attributes.items():
                setattr(db_char, key, value)

            char_triggers_texts = [trigger_text.text for trigger_text in db_char.trigger_texts]
            for trigger in triggers:
                if trigger not in char_triggers_texts:
                    db_char.trigger_texts.append(CharacterTriggerText(text=trigger))
            character_dict[request_char.name] = db_char

    session.add_all(to_update)
    session.commit()
    return character_dict


def get_character_from_names(
    messages_names: list[str], chat_id: int, similarity_dict: dict[str, str], session: Session
) -> dict[str, Character]:
    char_names = list(set(messages_names))
    db_characters = get_characters(char_names, chat_id, session)

    mapped_entities = [similarity_dict[name] for name in char_names]
    trigger_strings = []
    for entity_name in mapped_entities:
        entity_trigger_strings = [
            trigger_string for trigger_string, ent in similarity_dict.items() if ent == entity_name
        ]
        trigger_strings.append(entity_trigger_strings)

    to_update = []
    character_dict = {}
    for char_name, db_char, triggers in zip(char_names, db_characters, trigger_strings, strict=True):
        if db_char is None:
            new_char = Character(
                chat_id=chat_id,
                name=char_name,
            )

            for trigger in triggers:
                new_char.trigger_texts.append(CharacterTriggerText(text=trigger))
            to_update.append(new_char)
            character_dict[char_name] = new_char
        else:
            char_triggers_texts = [trigger_text.text for trigger_text in db_char.trigger_texts]
            for trigger in triggers:
                if trigger not in char_triggers_texts:
                    db_char.trigger_texts.append(CharacterTriggerText(text=trigger))
            character_dict[char_name] = db_char

    session.add_all(to_update)
    session.commit()
    return character_dict


def process_request(
    external_chat_id: str,
    chat_texts: list[str],
    messages_external_ids: list[str],
    messages_names: list[str],
    db_session,
    characters: list[ChatDataCharacter] | None = None,
    include_names: bool = False,
    external_user_id: str | None = None,
    token_limit: int | None = None,
) -> list[KnowledgeData]:
    start_time = timeit.default_timer()
    excluded_messages = 4

    user = get_user(external_user_id, db_session)
    chat = get_chat(user, external_chat_id, chat_texts, messages_external_ids, db_session)

    external_message_map = {}

    if settings.secondary_database.enabled:
        external_message_map = dict(zip(messages_external_ids, chat_texts, strict=True))

    doc_time = timeit.default_timer()
    entity_list, entity_dict = get_named_entities(chat_texts, messages_external_ids, messages_names, chat)
    doc_end_time = timeit.default_timer()
    general_logger.debug(f"Getting named entities {doc_end_time - doc_time} seconds")

    embedding_time = timeit.default_timer()
    embedding_dict = get_embeddings(chat, chat_texts, messages_names, external_message_map)
    embedding_time_end = timeit.default_timer()
    general_logger.debug(f"Getting vector embeddings {embedding_time_end - embedding_time} seconds")

    unique_ents: list[NamedEntity] = list(set(chain(*entity_list)))
    unique_ent_names = list({ent.name for ent in unique_ents})
    entity_similarity_dict = filter_similar_entities(unique_ent_names)

    last_messages = chat_texts[:-excluded_messages]  # exclude last few messages from saving
    last_names = messages_names[:-excluded_messages]
    last_external_ids = messages_external_ids[:-excluded_messages]
    last_entities = entity_list[:-excluded_messages]

    if characters:
        characters_dict = update_characters(characters, chat.id, entity_similarity_dict, db_session)
    else:
        characters_dict = get_character_from_names(messages_names, chat.id, entity_similarity_dict, db_session)

    new_message_indices, chat, new_messages = save_messages(
        last_messages, last_external_ids, last_names, characters_dict, entity_dict, embedding_dict, chat, db_session
    )

    knowledge_dict, entity_db_map = save_named_entities(chat, last_entities, entity_similarity_dict, db_session)
    link_knowledge(new_messages, knowledge_dict, entity_db_map, entity_similarity_dict, db_session)

    messages_to_summarize = [last_entities[index] for index in new_message_indices]

    ent_names = set()

    for entities in messages_to_summarize:
        for entity in set(entities):
            ent_names.add(entity.name)
            general_logger.debug(f"{entity.name}, {entity.label}, {spacy.explain(entity.label)}")

    to_summarize = list({entity_similarity_dict[ent_name] for ent_name in ent_names})
    for ent in to_summarize:
        summarize.delay(
            ent,
            chat.id,
            include_names,
        )

    vector_embeddings = np.array([embedding_dict[mes] for mes in chat_texts])
    ordered_knowledge = semantic_search(vector_embeddings, chat.id, db_session)

    knowledge_data = []
    if token_limit:
        current_tokens = 0
        for index, knowledge in enumerate(ordered_knowledge, 1):
            current_tokens += knowledge.token_count
            if current_tokens < token_limit:
                knowledge_data.append(KnowledgeData(text=knowledge.summary_entry, relevance=index))
            else:
                break
    else:
        knowledge_data = [
            KnowledgeData(text=knowledge.summary_entry, relevance=index)
            for index, knowledge in enumerate(ordered_knowledge, 1)
        ]

    end_time = timeit.default_timer()
    general_logger.info(f"Request processing time: {end_time - start_time}s")
    return knowledge_data
