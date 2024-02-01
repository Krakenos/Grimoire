import logging

from memoir.core.settings import DEBUG, LOG_PROMPTS

if DEBUG:
    LOG_LEVEL = logging.DEBUG
else:
    LOG_LEVEL = logging.INFO

if LOG_PROMPTS:
    PROMPT_LEVEL = logging.DEBUG
else:
    PROMPT_LEVEL = logging.INFO

logging.basicConfig(format='[%(asctime)s] %(levelname)s [%(module)s:%(funcName)s:%(lineno)d]: %(message)s')
formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(module)s:%(funcName)s:%(lineno)d]: %(message)s')

GENERAL_LOG_FILE = '../../general.log'
SUMMARY_LOG_FILE = '../../summary.log'
CONTEXT_LOG_FILE = '../../context.log'

general_logger = logging.getLogger('general')
summary_logger = logging.getLogger('summary')
context_logger = logging.getLogger('context')

general_logger.setLevel(LOG_LEVEL)
summary_logger.setLevel(PROMPT_LEVEL)
context_logger.setLevel(PROMPT_LEVEL)

general_file_handler = logging.FileHandler(GENERAL_LOG_FILE)
summary_file_handler = logging.FileHandler(SUMMARY_LOG_FILE)
context_file_handler = logging.FileHandler(CONTEXT_LOG_FILE)

general_file_handler.setFormatter(formatter)
summary_file_handler.setFormatter(formatter)
context_file_handler.setFormatter(formatter)

general_logger.addHandler(general_file_handler)
summary_logger.addHandler(summary_file_handler)
context_logger.addHandler(context_file_handler)

general_logger.addHandler(logging.StreamHandler())
summary_logger.addHandler(logging.StreamHandler())
context_logger.addHandler(logging.StreamHandler())
