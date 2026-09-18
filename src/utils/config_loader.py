from pathlib import Path

import yaml

from src.utils.shared_config import validate_search_options


def load_config(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f'Configuration must be a YAML mapping: {path}')
    return config


def validate_baseline_config(config: dict) -> None:
    for section in ('model', 'fire', 'search'):
        if not isinstance(config.get(section), dict):
            raise ValueError(f'Configuration section {section!r} must be a mapping.')
    model_name = config['model'].get('name')
    if not isinstance(model_name, str) or ':' not in model_name:
        raise ValueError('model.name must use provider:model_id format.')
    for section, key, minimum in (
        ('fire', 'max_steps', 0),
        ('fire', 'max_retries', 0),
        ('fire', 'max_tolerance', 2),
        ('search', 'num_searches', 1),
    ):
        value = config[section].get(key)
        if type(value) is not int or value < minimum:
            raise ValueError(f'{section}.{key} must be an integer >= {minimum}.')
    if not isinstance(config['fire'].get('diverse_prompt'), bool):
        raise ValueError('fire.diverse_prompt must be a boolean.')
    validate_search_options(config['search'].get('type'), config['search']['num_searches'])
