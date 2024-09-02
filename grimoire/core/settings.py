import os
import pathlib

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class SecondaryDatabaseSettings(BaseModel):
    enabled: bool = False
    db_engine: str = ""
    message_encryption: str = "aesgcm"
    encryption_key: str = ""


class TokenizationSettings(BaseModel):
    prefer_local_tokenizer: bool = True
    local_tokenizer: str = "oobabooga/llama-tokenizer"


class SummarizationSettings(BaseModel):
    prompt: str = (
        "{system_sequence}{previous_summary}{messages}{system_suffix}\n"
        "{input_sequence}Describe {term}.{input_suffix}{output_sequence}"
    )
    limit_rate: int = 1
    bos_token: str = "<s>"
    max_tokens: int = 300
    params: dict = {"min_p": 0.1, "rep_pen": 1.0, "temperature": 0.6, "stop": ["</s>"], "stop_sequence": ["</s>"]}


class ApiSettings(BaseModel):
    backend: str = "GenericOAI"
    url: str = ""
    auth_key: str = ""
    context_length: int = 4096
    system_sequence: str = ""
    system_suffix: str = ""
    input_sequence: str = "### Instruction:\n"
    input_suffix: str = "\n"
    output_sequence: str = "### Response:\n"
    output_suffix: str = "\n"
    first_output_sequence: str = ""
    last_output_sequence: str = ""


class Settings(BaseModel):
    REDIS_HOST: str = "127.0.0.1"
    REDIS_PORT: str | int = 6379
    REDIS_SENTINEL: bool = False
    SENTINEL_MASTER_NAME: str = "mymaster"
    REDIS_TLS: bool = False
    CACHE_EXPIRE_TIME: int = 86400
    DB_ENGINE: str = "postgresql+psycopg2://grimoire:secretpassword@127.0.0.1:5432/grimoire"
    DEBUG: bool = False
    LOG_PROMPTS: bool = False
    LOG_FILES: bool = False
    AUTH_KEY: str | None = None
    ENCRYPTION_KEY: str = "sample-database-encryption-key"
    prefer_gpu: bool = False
    match_distance: int = 80
    summarization_api: ApiSettings = ApiSettings()
    summarization: SummarizationSettings = SummarizationSettings()
    tokenization: TokenizationSettings = TokenizationSettings()
    secondary_database: SecondaryDatabaseSettings = SecondaryDatabaseSettings()


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


loaded_settings = SettingsLoader.load_config()
settings = Settings(**loaded_settings)
