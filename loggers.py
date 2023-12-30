import logging

LOG_LEVEL = logging.DEBUG

GENERAL_LOG_FILE = 'general.log'
SUMMARY_LOG_FILE = 'summary.log'
CONTEXT_LOG_FILE = 'context.log'

general_logger = logging.getLogger('general')
summary_logger = logging.getLogger('summary')
context_logger = logging.getLogger('context')

general_logger.setLevel(LOG_LEVEL)
summary_logger.setLevel(LOG_LEVEL)
context_logger.setLevel(LOG_LEVEL)

general_logger.addHandler(logging.FileHandler(GENERAL_LOG_FILE))
summary_logger.addHandler(logging.FileHandler(SUMMARY_LOG_FILE))
context_logger.addHandler(logging.FileHandler(CONTEXT_LOG_FILE))

general_logger.addHandler(logging.StreamHandler())
summary_logger.addHandler(logging.StreamHandler())
context_logger.addHandler(logging.StreamHandler())
