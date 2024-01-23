import requests

from memoir.core.settings import MAIN_API_URL


def get_passthrough(endpoint: str, auth_token=None) -> dict:
    passthrough_url = MAIN_API_URL + endpoint
    kobold_response = requests.get(passthrough_url, headers={'Authorization': f'Bearer {auth_token}'})
    return kobold_response.json()


def orm_get_or_create(session, db_model, **kwargs):
    instance = session.query(db_model).filter_by(**kwargs).first()
    if instance:
        return instance
    else:
        instance = db_model(**kwargs)
        session.add(instance)
        session.commit()
        return instance
