import numpy as np
from rapidfuzz import fuzz, process, utils
from sentence_transformers.util import cos_sim
from spacy.matcher.dependencymatcher import defaultdict
from sqlalchemy import select
from sqlalchemy.orm import Session
from itertools import combinations

from grimoire.common.utils import time_execution
from grimoire.core.settings import settings
from grimoire.db.models import Character, Knowledge, Message
import networkx as nx


def get_knowledge_entity(term: str, chat_id: int, session: Session) -> Knowledge | None:
    return get_knowledge_entities([term], chat_id, session)[0]


def get_knowledge_entities(terms: list[str], chat_id: int, session: Session) -> list[Knowledge | None]:
    query = select(Knowledge).where(Knowledge.entity_type == "NAMED ENTITY", Knowledge.chat_id == chat_id)
    query_results = session.scalars(query).all()
    results = []
    knowledge_dict = {knowledge_obj.entity: knowledge_obj for knowledge_obj in query_results}
    for term in terms:
        found_entries = process.extract(
            term,
            list(knowledge_dict),
            scorer=fuzz.partial_ratio,
            processor=utils.default_process,
            score_cutoff=settings.match_distance,
        )
        if not found_entries:
            results.append(None)
        else:
            max_score = max([entry[1] for entry in found_entries])
            max_entries = list(filter(lambda x: x[1] == max_score, found_entries))
            if len(max_entries) == 1:
                ent_name, _, _ = max_entries[0]
            else:
                # If entries have the same score use regular ratio as tiebreaker
                max_entry_names = [entry[0] for entry in max_entries]
                best_match = process.extractOne(
                    term, max_entry_names, scorer=fuzz.ratio, processor=utils.default_process
                )
                ent_name, _, _ = best_match
            results.append(knowledge_dict[ent_name])
    return results


@time_execution
def semantic_search(message_embeddings: np.ndarray, chat_id: int, session: Session) -> list[Knowledge]:
    message_amount = len(message_embeddings)
    weights = np.linspace(0.5, 1, num=message_amount)
    candidates = []

    for embedding in message_embeddings:
        query = (
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.vector_embedding.cosine_distance(embedding))
            .limit(3)
        )
        similar_messages = session.scalars(query)
        for message in similar_messages:
            candidates.extend(message.knowledge)

    candidates = list(set(candidates))
    candidates = [candidate for candidate in candidates if candidate.vector_embedding is not None and candidate.enabled]

    if not candidates:
        return []

    # Cast as float64
    candidates_embeddings = np.array([candidate.vector_embedding for candidate in candidates], dtype="float64")
    message_embeddings = message_embeddings.astype("float64")

    cosine_similarity = cos_sim(message_embeddings, candidates_embeddings)
    weighted_similarity = cosine_similarity.T * weights
    highest_similarities = np.max(weighted_similarity.numpy(), axis=1)
    sorted_indices = np.argsort(highest_similarities)[::-1]
    sorted_knowledge = [candidates[i] for i in sorted_indices]
    return sorted_knowledge


def get_character(name: str, chat_id: int, session: Session) -> Character | None:
    return get_characters([name], chat_id, session)[0]


def get_characters(names: list[str], chat_id: int, session: Session) -> list[Character | None]:
    query = select(Character).where(Character.name.in_(names), Character.chat_id == chat_id)
    query_results = session.scalars(query).all()
    db_chars = {char.name: char for char in query_results}
    results = [db_chars[name] if name in db_chars else None for name in names]
    return results


def get_messages_by_index(start_index: int, end_index: int, chat_id: int, session: Session) -> list[Message]:
    query = (
        select(Message)
        .where(Message.chat_id == chat_id, Message.message_index >= start_index, Message.message_index <= end_index)
        .order_by(Message.message_index)
    )
    query_results = session.scalars(query).all()
    return list(query_results)


def get_knowledge_graph(chat_id: int, session: Session):
    query = select(Knowledge).where(Knowledge.chat_id == chat_id).order_by(Knowledge.id)
    knowledge_entries = list(session.scalars(query).all())
    knowledge_dict = {knowledge.id: knowledge for knowledge in knowledge_entries}
    memory_dict = {}
    graph = nx.Graph()

    relations = defaultdict(set)
    for knowledge in knowledge_entries:
        graph.add_node(f"{knowledge.id} {knowledge.entity}")
        for mes in knowledge.messages:
            for memory in mes.segmented_memories:
                memory_dict[memory.id] = memory
                relations[memory.id].add(knowledge.id)

    for memory_id, knowledge_set in relations.items():
        for a, b in combinations(knowledge_set, 2):
            graph.add_edge(f"{a} {knowledge_dict[a].entity}", f"{b} {knowledge_dict[b].entity}", label=f"memory {memory_id}")
