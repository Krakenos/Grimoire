import copy
import os
import pathlib

import yaml
from dotenv import load_dotenv

from grimoire.core.default_settings import defaults

load_dotenv()


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

        default_settings_path = proj_path / "config" / "settings.yaml"

        config_path = os.environ.get('APP_CONFIG', default_settings_path.resolve())

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
    settings_dict = copy.deepcopy(settings_dict)
    for key, value in overrides.items():
        if key in settings_dict and value not in ('', None):
            match value:
                case dict():
                    settings_dict[key] = merge_settings(settings_dict[key], value)
                case _:
                    settings_dict[key] = value
    return settings_dict


settings = copy.deepcopy(defaults)
loaded_settings = SettingsLoader.load_config()
settings = merge_settings(settings, loaded_settings)

if settings['side_api']['url'] in ('', None):
    settings['single_api_mode'] = True
