from urllib.parse import urljoin

import requests

from grimoire.core.settings import settings


def get_passthrough(endpoint: str, auth_token=None) -> dict:

    passthrough_url = urljoin(settings['main_api']['url'], endpoint)
    engine = requests.get(passthrough_url, headers={'Authorization': f'Bearer {auth_token}'})
    return engine.json()


def orm_get_or_create(session, db_model, **kwargs):
    instance = session.query(db_model).filter_by(**kwargs).first()
    if instance:
        return instance
    else:
        instance = db_model(**kwargs)
        session.add(instance)
        session.commit()
        return instance
