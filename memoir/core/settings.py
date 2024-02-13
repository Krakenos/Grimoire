import os
import pathlib

import yaml
from dotenv import load_dotenv

from memoir.core.default_settings import defaults

load_dotenv()

SINGLE_API_MODE = bool(os.getenv('SINGLE_API_MODE', False))
CONTEXT_PERCENTAGE = float(os.getenv('CONTEXT_PERCENTAGE'))

MAIN_API_BACKEND = os.getenv('MAIN_API_BACKEND', 'GenericOAI')
MAIN_API_URL = os.getenv('MAIN_API_URL')
MAIN_API_AUTH = os.getenv('MAIN_API_AUTH')

SIDE_API_BACKEND = os.getenv('SIDE_API_BACKEND', 'GenericOAI')
SIDE_API_URL = os.getenv('SIDE_API_URL')
SIDE_API_AUTH = os.getenv('SIDE_API_AUTH')

CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL')
DB_ENGINE = os.getenv('DB_ENGINE')

DEBUG = bool(os.getenv('DEBUG', False))
LOG_PROMPTS = bool(os.getenv('LOG_PROMPTS', False))

MODEL_INPUT_SEQUENCE = '### Instruction:\n'
MODEL_OUTPUT_SEQUENCE = '\n### Response:\n'

SUMMARIZATION_PROMPT = '{start_token}{previous_summary}\n{messages}\n{input_sequence}Describe {term}.{output_sequence}'
SUMMARIZATION_INPUT_SEQ = '### Instruction:\n'
SUMMARIZATION_OUTPUT_SEQ = '\n### Response:\n'
SUMMARIZATION_START_TOKEN = '<s>'
SUMMARIZATION_PARAMS = {
    "min_p": 0.1,
    "rep_pen": 1.0,
    "temperature": 0.6,
    "stop": [
        "</s>"
    ],
    "stop_sequence": [
        "</s>"
    ]
}

if SIDE_API_URL == '' or SIDE_API_URL is None:
    SINGLE_API_MODE = True


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
        settings_file_name = os.environ.get('SETTINGS_FILE_NAME', 'settings.yaml')
        config_path = os.environ.get('APP_CONFIG', settings_file_name)
        return config_path

    @classmethod
    def load_from_file(cls, file_path: str) -> dict:
        loader = cls.make_config_loader()
        with open(file_path) as f:
            return yaml.load(stream=f.read(), Loader=loader)

    @classmethod
    def make_config_loader(cls):
        loader = yaml.SafeLoader
        loader.add_constructor('!env', envvar_constructor)
        return loader

    @classmethod
    def load_config(cls) -> dict:
        path = cls.settings_path()
        return cls.load_from_file(path)


def merge_settings(settings_dict, overrides):
    settings_dict = settings_dict.copy()
    for key, value in overrides.items():
        if key in settings_dict and value not in ('', None):
            match value:
                case dict():
                    settings_dict[key] = merge_settings(settings_dict[key], value)
                case _:
                    settings_dict[key] = value
    return settings_dict


settings = defaults.copy()
loaded_settings = SettingsLoader.load_config()
settings = merge_settings(settings, loaded_settings)
