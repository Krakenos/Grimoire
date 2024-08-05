from sqlalchemy import select
from sqlalchemy.orm import Session

from grimoire.db.models import Knowledge


def get_knowledge_entity(term: str, chat_id: int, session: Session) -> Knowledge | None:
    query = select(Knowledge).where(Knowledge.entity_type == "NAMED ENTITY", Knowledge.chat_id == chat_id)
    query_results = session.scalars(query).all()
    knowledge_dict = {knowledge_obj.entity.lower(): knowledge_obj for knowledge_obj in query_results}
    if term.lower() in knowledge_dict:
        return knowledge_dict[term.lower()]
    return None


def get_knowledge_entities(terms: list[str], chat_id: int, session: Session) -> list[Knowledge]:
    query = select(Knowledge).where(Knowledge.entity_type == "NAMED ENTITY", Knowledge.chat_id == chat_id)
    query_results = session.scalars(query).all()
    knowledge_dict = {knowledge_obj.entity.lower(): knowledge_obj for knowledge_obj in query_results}
    results = [knowledge_dict[term.lower()] for term in terms if term.lower() in knowledge_dict]
    return results
