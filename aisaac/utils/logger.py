import logging
import logging.config
import sys

class Logger:
    def __init__(self, name, context_manager=None, level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        if not context_manager:
            self.level = level
        else:
            self.level = context_manager.get_config('LOGGING_LEVEL')
        self.debug_log_file = "debug.log"
        self.error_log_file = "error.log"
        self.setup_logging()

    def setup_logging(self):
        handlers_config = {
            "stdout": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "stream": sys.stdout
            }
        }

        if self.debug_log_file:  # Debug file handler
            handlers_config["debug_file"] = {
                "class": "logging.FileHandler",
                "formatter": "simple",
                "filename": self.debug_log_file,
                "mode": 'a',  # Append mode
                "level": logging.DEBUG
            }

        if self.error_log_file:  # Error file handler
            handlers_config["error_file"] = {
                "class": "logging.FileHandler",
                "formatter": "simple",
                "filename": self.error_log_file,
                "mode": 'a',  # Append mode
                "level": logging.ERROR
            }

        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "simple": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s: %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                }
            },
            "handlers": handlers_config,
            "loggers": {
                self.logger.name: {
                    "level": self.level,
                    "handlers": list(handlers_config.keys())  # Use all configured handlers
                }
            }
        }
        logging.config.dictConfig(logging_config)
        self.logger.setLevel(self.level)

    def get_logger(self):
        return self.logger
