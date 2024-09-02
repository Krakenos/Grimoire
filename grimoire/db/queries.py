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
        found_entry = process.extractOne(
            term,
            list(knowledge_dict),
            scorer=fuzz.WRatio,
            processor=utils.default_process,
            score_cutoff=settings.match_distance,
        )
        if found_entry is None:
            results.append(None)
        else:
            ent_name, _, _ = found_entry
            results.append(knowledge_dict[ent_name])
    return results
