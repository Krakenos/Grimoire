defaults = {
    'CELERY_BROKER_URL': 'redis://localhost:6379/0',
    'DB_ENGINE': 'sqlite:///db.sqlite3',
    'DEBUG': False,
    'LOG_PROMPTS': False,
    'single_api_mode': False,
    'context_percentage': 0.25,
    'main_api': {
        'backend': 'GenericOAI',
        'url': '',
        'auth_key': '',
        'input_sequence': '### Instruction:\n',
        'output_sequence': '\n### Response:\n'
    },
    'side_api': {
        'backend': 'GenericOAI',
        'url': '',
        'auth_key': '',
        'input_sequence': '### Instruction:\n',
        'output_sequence': '\n### Response:\n'
    },
    'summarization': {
        'prompt': '{bos_token}{previous_summary}\n{messages}\n{input_sequence}Describe {term}.{output_sequence}',
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
