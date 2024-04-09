defaults = {
    'CELERY_BROKER_URL': 'redis://127.0.0.1:6379/0',
    'DB_ENGINE': 'postgresql+psycopg2://grimoire:secretpassword@127.0.0.1:5432/grimoire',
    'DEBUG': False,
    'LOG_PROMPTS': False,
    'single_api_mode': False,
    'multi_user_mode': False,
    'prefer_gpu': False,
    'context_percentage': 0.25,
    'preserved_messages': 2,
    'main_api': {
        'backend': 'GenericOAI',
        'url': '',
        'auth_key': '',
        'system_sequence': '',
        'system_suffix': '',
        'input_sequence': '### Instruction:\n',
        'input_suffix': '',
        'output_sequence': '\n### Response:\n',
        'output_suffix': '',
        'first_output_sequence': '',
        'last_output_sequence': '',
        'wrap': False,
        'collapse_newlines': False
    },
    'side_api': {
        'backend': 'GenericOAI',
        'url': '',
        'auth_key': '',
        'system_sequence': '',
        'system_suffix': '',
        'input_sequence': '### Instruction:\n',
        'input_suffix': '',
        'output_sequence': '\n### Response:\n',
        'output_suffix': '',
        'first_output_sequence': '',
        'last_output_sequence': '',
        'wrap': False,
        'collapse_newlines': False
    },
    'summarization': {
        'prompt': '{bos_token}{previous_summary}\n{messages}\n{input_sequence}Describe {term}.{output_sequence}',
        'limit_rate': 1,
        'bos_token': '<s>',
        'params': {
            'min_p': 0.1,
            'rep_pen': 1.0,
            'temperature': 0.6,
            'stop': [
                '</s>'
            ],
            "stop_sequence": [
                '</s>'
            ]
        }
    }
}
