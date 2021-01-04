import logging
import os

import config

logging_level = getattr(logging, config.LOGGING_LEVEL)
logging.basicConfig(format=config.LOGGING_FORMAT, level=logging_level)

# implicit library optimizations
os.environ["OPENBLAS_NUM_THREADS"] = "1"
