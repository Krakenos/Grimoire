import copy
import os
import pathlib

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from grimoire.core.default_settings import defaults

load_dotenv()


# TODO Move defaults to pydantic model and refactor settings dict to pydantic object
class SecondaryDatabaseSettingsValidator(BaseModel):
    enabled: bool | None = None
    db_engine: str | None = None
    message_encryption: str | None = None
    encryption_key: str | None = None


class TokenizationSettingsValidator(BaseModel):
    prefer_local_tokenizer: bool | None = None
    local_tokenizer: str | None = None


class SummarizationSettingsValidator(BaseModel):
    prompt: str | None = None
    limit_rate: int | None = None
    bos_token: str | None = None
    max_tokens: int | None = None
    params: dict | None = None


class ApiSettingsValidator(BaseModel):
    backend: str | None = None
    url: str | None = None
    auth_key: str | None = None
    context_length: int | None = None
    system_sequence: str | None = None
    system_suffix: str | None = None
    input_sequence: str | None = None
    input_suffix: str | None = None
    output_sequence: str | None = None
    output_suffix: str | None = None
    first_output_sequence: str | None = None
    last_output_sequence: str | None = None


class SettingsValidator(BaseModel):
    REDIS_HOST: str | None = None
    REDIS_PORT: str | int | None = None
    REDIS_SENTINEL: bool | None = None
    SENTINEL_MASTER_NAME: str | None
    REDIS_TLS: bool | None = None
    CACHE_EXPIRE_TIME: int | None = None
    DB_ENGINE: str | None = None
    DEBUG: bool | None = None
    LOG_PROMPTS: bool | None = None
    LOG_FILES: bool | None = None
    AUTH_KEY: str | None = None
    ENCRYPTION_KEY: str | None = None
    prefer_gpu: bool | None = None
    match_distance: int | None = None
    summarization_api: ApiSettingsValidator | None = None
    summarization: SummarizationSettingsValidator | None = None
    tokenization: TokenizationSettingsValidator | None = None
    secondary_database: SecondaryDatabaseSettingsValidator | None = None


def envvar_constructor(loader: yaml.Loader, node: yaml.ScalarNode):
    """
    Extracts the environment variable from the node's value
    :param yaml.Loader loader: the yaml loader
    :param node: the current node in the yaml
    :return: the parsed string that contains the value of the environment
    variable
    """
    value = loader.construct_scalar(node)
    value = os.environ.get(value, None)
    return value


class SettingsLoader:
    @classmethod
    def settings_path(cls):
        proj_dir = os.environ.get("PYTHONPATH")

        if proj_dir and proj_dir.startswith("/app"):  # Dockerfile
            proj_path = pathlib.Path(proj_dir)
        else:  # Other envs
            proj_path = pathlib.Path(__file__).parents[2]

        settings_file = os.environ.get("SETTINGS_FILE", "settings.yaml")

        settings_path = proj_path / "config" / settings_file

        return settings_path.resolve()

    @classmethod
    def load_from_file(cls, file_path: str) -> dict:
        loader = cls.make_config_loader()
        with open(file_path) as f:
            return yaml.load(stream=f.read(), Loader=loader)

    @classmethod
    def make_config_loader(cls):
        loader = yaml.SafeLoader
        loader.add_constructor("!env", envvar_constructor)
        return loader

    @classmethod
    def load_config(cls) -> dict:
        path = cls.settings_path()
        return cls.load_from_file(path)


def merge_settings(settings_dict, overrides):
    settings_dict = copy.deepcopy(settings_dict)
    for key, value in overrides.items():
        if key in settings_dict and value not in ("", None):
            match value:
                case dict():
                    settings_dict[key] = merge_settings(settings_dict[key], value)
                case _:
                    settings_dict[key] = value
    return settings_dict


settings = copy.deepcopy(defaults)
loaded_settings = SettingsLoader.load_config()
validated_settings = SettingsValidator(**loaded_settings).model_dump()
settings = merge_settings(settings, validated_settings)
