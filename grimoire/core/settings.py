import json
import os
import pathlib
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator, model_validator
from pydantic_core.core_schema import ValidationInfo

load_dotenv()


class BaseSettingsModel(BaseModel):
    @field_validator("*", mode="before")
    @classmethod
    def replace_none(cls, v: Any, info: ValidationInfo) -> Any:
        if v is None:
            return cls.model_fields[info.field_name].default
        return v


class SecondaryDatabaseSettings(BaseSettingsModel):
    enabled: bool = False
    db_engine: str = ""
    message_encryption: str = "aesgcm"
    encryption_key: str = ""


class TokenizationSettings(BaseSettingsModel):
    prefer_local_tokenizer: bool = True
    local_tokenizer: str = "oobabooga/llama-tokenizer"


class SummarizationSettings(BaseSettingsModel):
    prompt: str = (
        "{system_sequence}{previous_summary}{additional_info}{messages}{system_suffix}\n"
        "{input_sequence}Describe {term}.{input_suffix}{output_sequence}"
    )
    segmented_memory_prompt: str = (
        "{system_sequence}Below is conversation snippet.\n{messages}{system_suffix}{input_sequence}"
        "Summarize the most important facts and events in the story so far. Limit the summary to one paragraph. "
        "Your response should include nothing but the summary.{input_suffix}{output_sequence}"
    )
    limit_rate: int = 1
    max_tokens: int = 300
    params: dict = {"min_p": 0.1, "rep_pen": 1.0, "temperature": 0.6, "stop": ["</s>"], "stop_sequence": ["</s>"]}

    @field_validator("params")
    @classmethod
    def add_stop(cls, v: dict) -> dict:
        if "stop" not in v.keys():
            v["stop"] = []

        if "stop_sequence" not in v.keys():
            v["stop_sequence"] = []

        return v

    @field_validator("prompt")
    @classmethod
    def replace_newline(cls, v: str) -> str:
        v = v.replace("\\n", "\n")
        return v

    @field_validator("params", mode="before")
    @classmethod
    def parse_string(cls, v: Any) -> Any:
        if isinstance(v, str):
            return json.loads(v)
        return v


class ApiSettings(BaseSettingsModel):
    backend: str = "GenericOAI"
    model: str = ""
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
    bos_token: str = "<s>"

    @field_validator(
        "system_sequence", "system_suffix", "input_sequence", "input_suffix", "output_sequence", "output_suffix"
    )
    @classmethod
    def replace_newline(cls, v: str) -> str:
        v = v.replace("\\n", "\n")
        return v


class RedisSettings(BaseSettingsModel):
    HOST: list[tuple[str, int]] = [("127.0.0.1", 6370)]
    SENTINEL: bool = False
    TLS: bool = False
    SENTINEL_MASTER_NAME: str = "mymaster"
    CACHE_EXPIRE_TIME: int = 86400

    @field_validator("HOST", mode="before")
    @classmethod
    def parse_string(cls, v: Any) -> Any:
        if isinstance(v, str):
            host_list = []

            for full_address in v.split(","):
                split_address = full_address.split(":")
                address = split_address[0]
                port = int(split_address[1])
                host_list.append((address, int(port)))

            return host_list
        return v


class Settings(BaseSettingsModel):
    DB_ENGINE: str = "postgresql+psycopg2://grimoire:secretpassword@127.0.0.1:5432/grimoire"
    DEBUG: bool = False
    LOG_PROMPTS: bool = False
    LOG_FILES: bool = False
    enable_management_panel: bool = False
    AUTH_KEY: str | None = None
    # Optional bearer key for the management panel's API; unset leaves the panel open, which is
    # fine only because the panel is meant to be local (see grimoire/api/auth.py:check_panel_key).
    # Must not be AUTH_KEY, which is handed out to chat clients.
    PANEL_KEY: str | None = None
    ENCRYPTION_KEY: str = "sample-database-encryption-key"
    HF_TOKEN: str | None = None
    EMBEDDING_MODEL: str = "Alibaba-NLP/gte-base-en-v1.5"
    # Pinned to a specific commit for security (avoids pulling unreviewed remote code via
    # trust_remote_code). This is also the fallback when EMBEDDING_MODEL_REVISION is unset in
    # the env-mapped config path, so the revision never silently floats to the latest.
    EMBEDDING_MODEL_REVISION: str | None = "a8e4f3e0ee719c75bc30d12b8eae0f8440502718"
    # The auto_map remote code (modeling.py) is hosted in a separate repo (Alibaba-NLP/new-impl)
    # that EMBEDDING_MODEL_REVISION does not cover. Pin the code commit so the executed
    # trust_remote_code never floats to that repo's latest main.
    EMBEDDING_MODEL_CODE_REVISION: str | None = "40ced75c3017eb27626c9d4ea981bde21a2662f4"
    prefer_gpu: bool = False
    match_distance: int = 80
    match_distance_short: int = 95
    # Origins allowed to make cross-origin requests (e.g. the SillyTavern browser tab).
    # Set to ["*"] to allow all origins, or list specific origins for better security.
    CORS_ALLOW_ORIGINS: list[str] = ["http://127.0.0.1:8000", "http://localhost:8000"]
    redis: RedisSettings = RedisSettings()
    summarization_api: ApiSettings = ApiSettings()
    summarization: SummarizationSettings = SummarizationSettings()
    tokenization: TokenizationSettings = TokenizationSettings()
    secondary_database: SecondaryDatabaseSettings = SecondaryDatabaseSettings()

    @field_validator("CORS_ALLOW_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @model_validator(mode="after")
    def check_panel_key_distinct(self) -> "Settings":
        """Reject PANEL_KEY reusing AUTH_KEY.

        PANEL_KEY is optional — the panel is a local tool and needs no key on a machine only you
        can reach. But setting it to AUTH_KEY is never what anyone wants: AUTH_KEY is handed to
        every chat client, so it would turn each client key into a key over every user's data.
        """
        if self.enable_management_panel and self.AUTH_KEY and self.PANEL_KEY == self.AUTH_KEY:
            raise ValueError(
                "PANEL_KEY must differ from AUTH_KEY. AUTH_KEY is distributed to chat clients, so "
                "reusing it would give every client full read/write access to the panel."
            )
        return self


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
