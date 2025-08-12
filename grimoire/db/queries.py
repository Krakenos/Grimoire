from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np
from rapidfuzz import fuzz, process, utils
from sentence_transformers.util import cos_sim
from sqlalchemy import select
from sqlalchemy.orm import Session

from grimoire.common.utils import time_execution
from grimoire.core.settings import settings
from grimoire.db.models import Character, Chat, Knowledge, Message, SegmentedMemory


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
def semantic_search(
    message_embeddings: np.ndarray, chat_id: int, session: Session
) -> list[Knowledge | SegmentedMemory]:
    message_amount = len(message_embeddings)
    weights = np.linspace(0.5, 1, num=message_amount)
    knowledge_candidates = []
    memory_candidates = []

    for embedding in message_embeddings:
        query = (
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.vector_embedding.cosine_distance(embedding))
            .limit(3)
        )
        similar_messages = session.scalars(query)
        for message in similar_messages:
            knowledge_candidates.extend(message.knowledge)
            memory_candidates.extend(message.segmented_memories)

    knowledge_candidates = list(set(knowledge_candidates))
    knowledge_candidates = [
        candidate for candidate in knowledge_candidates if candidate.vector_embedding is not None and candidate.enabled
    ]
    memory_candidates = list(set(memory_candidates))
    memory_candidates = [candidate for candidate in memory_candidates if candidate.vector_embedding is not None]
    all_candidates: list[Knowledge | SegmentedMemory] = [*knowledge_candidates, *memory_candidates]

    if not all_candidates:
        return []

    # Cast as float64
    candidates_embeddings = np.array([candidate.vector_embedding for candidate in all_candidates], dtype="float64")
    message_embeddings = message_embeddings.astype("float64")

    cosine_similarity = cos_sim(message_embeddings, candidates_embeddings)
    weighted_similarity = cosine_similarity.T * weights
    highest_similarities = np.max(weighted_similarity.numpy(), axis=1)
    sorted_indices = np.argsort(highest_similarities)[::-1]
    sorted_entries = [all_candidates[i] for i in sorted_indices]
    return sorted_entries


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


def get_knowledge_graph(chat_id: int, user_id: int, session: Session) -> nx.Graph:
    query = (
        select(Knowledge).join(Chat).where(Knowledge.chat_id == chat_id, Chat.user_id == user_id).order_by(Knowledge.id)
    )
    knowledge_entries = list(session.scalars(query).all())
    graph = nx.Graph()

    relations = defaultdict(set)
    for knowledge in knowledge_entries:
        knowledge_memories = set()
        for mes in knowledge.messages:
            for memory in mes.segmented_memories:
                knowledge_memories.add(memory.id)
                relations[memory.id].add(str(knowledge.id))
        graph.add_node(str(knowledge.id), label=f"{knowledge.entity}", memory_list=sorted(knowledge_memories))

    links_amount = defaultdict(lambda: defaultdict(lambda: 0))
    links_memories = defaultdict(lambda: defaultdict(set))
    for memory_id, knowledge_set in relations.items():
        for a, b in combinations(knowledge_set, 2):
            links_amount[a][b] += 1
            links_memories[a][b].add(memory_id)

    for key, values in links_amount.items():
        for key2, connections in values.items():
            lab = "1 memory" if connections == 1 else f"{connections} memories"
            graph.add_edge(key, key2, weight=connections, label=lab, memory_list=sorted(links_memories[key][key2]))

    # Debug stuff for graphs
    # nt = Network("2000px", "2000px")
    # nt.from_nx(graph)
    # nt.write_html("graph.html")

    return graph
