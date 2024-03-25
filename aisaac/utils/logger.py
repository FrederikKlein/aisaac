import logging
import logging.config
import sys


class Logger:
    def __init__(self, name, context_manager, level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.level = context_manager.get_config('LOGGING_LEVEL')
        self.setup_logging()

    def setup_logging(self):
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "simple": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s: %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                }
            },
            "handlers": {
                "stdout": {
                    "class": "logging.StreamHandler",
                    "formatter": "simple",
                    "stream": sys.stdout
                }
            },
            "loggers": {
                self.logger.name: {
                    "level": self.level,
                    "handlers": ["stdout"]
                }
            }
        }
        logging.config.dictConfig(logging_config)
        self.logger.setLevel(self.level)

    def get_logger(self):
        return self.logger
