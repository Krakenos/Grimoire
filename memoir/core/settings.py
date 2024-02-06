import os

from dotenv import load_dotenv

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
