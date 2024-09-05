from rapidfuzz import fuzz, process, utils
from sqlalchemy import select
from sqlalchemy.orm import Session

from grimoire.core.settings import settings
from grimoire.db.models import Knowledge


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
                    term,
                    max_entry_names,
                    scorer=fuzz.ratio,
                    processor=utils.default_process
                )
                ent_name, _, _ = best_match
            results.append(knowledge_dict[ent_name])
    return results
