import timeit
from functools import wraps
from urllib.parse import urljoin

import requests

from grimoire.common.loggers import general_logger
from grimoire.core.settings import settings


def get_passthrough(endpoint: str, auth_token=None) -> dict:
    passthrough_url = urljoin(settings["main_api"]["url"], endpoint)
    engine = requests.get(passthrough_url, headers={"Authorization": f"Bearer {auth_token}"})
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


def time_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        general_logger.debug(f"Function {func.__name__} took {end_time - start_time} seconds to execute.")
        return result

    return wrapper


def async_time_execution(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = await func(*args, **kwargs)
        end_time = timeit.default_timer()
        general_logger.debug(f"Async function {func.__name__} took {end_time - start_time} seconds to execute.")
        return result

    return wrapper
