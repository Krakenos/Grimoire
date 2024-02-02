import os
from dotenv import load_dotenv

load_dotenv()

MAIN_API_BACKEND = 'Aphrodite'
MAIN_API_URL = os.getenv('MAIN_API_URL')
MAIN_API_AUTH = os.getenv('MAIN_API_AUTH')

SIDE_API_BACKEND = 'KoboldCpp'
SIDE_API_URL = os.getenv('SIDE_API_URL')
CONTEXT_PERCENTAGE = float(os.getenv('CONTEXT_PERCENTAGE'))

CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL')
DB_ENGINE = os.getenv('DB_ENGINE')

DEBUG = bool(os.getenv('DEBUG', False))
LOG_PROMPTS = bool(os.getenv('LOG_PROMPTS', False))

MODEL_INPUT_SEQUENCE = '### Instruction:'
MODEL_OUTPUT_SEQUENCE = '### Response:'

SUMMARIZATION_PROMPT = 'Based on following text describe {term}.'
SUMMARIZATION_INPUT_SEQ = '[INST]'
SUMMARIZATION_OUTPUT_SEQ = '[/INST]'
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
