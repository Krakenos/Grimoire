import os

from dotenv import load_dotenv

from memoir.common.loggers import general_logger

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

SUMMARIZATION_PROMPT = 'Based on following text describe {term}.'
SUMMARIZATION_INPUT_SEQ = '[INST] '
SUMMARIZATION_OUTPUT_SEQ = ' [/INST]'
SUMMARIZATION_START_TOKEN = '<s>'
SUMMARIZATION_PARAMS = {
    "top_p": 0.95,
    "top_k": 50,
    "rep_pen": 1.2,
    "temperature": 0.6,
    "stop": [
        "</s>"
    ],
    "stop_sequence": [
        "</s>"
    ]
}

if SIDE_API_URL == '' or SIDE_API_URL is None:
    general_logger.warning('SIDE_API_URL is not set, changing to single API mode')
    SINGLE_API_MODE = True
