import logging

import config

logging_level = getattr(logging, config.LOGGING_LEVEL)
logging.basicConfig(format=config.LOGGING_FORMAT, level=logging_level)
