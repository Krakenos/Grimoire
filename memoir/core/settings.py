import os

MAIN_API_URL = os.getenv('MAIN_API_URL')
MAIN_API_AUTH = os.getenv('MAIN_API_AUTH')
SIDE_API_URL = os.getenv('SIDE_API_URL')
CONTEXT_PERCENTAGE = float(os.getenv('CONTEXT_PERCENTAGE'))
DB_ENGINE = os.getenv('DB_ENGINE')
MODEL_INPUT_SEQUENCE = '### Instruction:'
MODEL_OUTPUT_SEQUENCE = '### Response:'
MAIN_API_BACKEND = 'Aphrodite'
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL')