import logging
import logging.config
import os


class CustomFormatter(logging.Formatter):
    green = "\x1b[32;20m"
    grey = "\x1b[37;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


app_name = os.environ.get("APP_NAME", "")


def get_logger(name: str):
    """Returns a logger with the given name, inheriting from the app's logger."""
    return logging.getLogger(app_name + "." + name)


# App loggers setup
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "custom": {
            "()": CustomFormatter,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "custom",
        },
    },
    "loggers": {
        app_name: {
            "level": os.getenv(f"{app_name.upper()}_LOG_LEVEL", logging.DEBUG),
        },
        # Application loggers
        f"{app_name}.plugins": {
            "level": "DEBUG",
        },
        # External loggers
        # "google_adk.google.adk.models.google_llm": {
        #     "level": "DEBUG",
        # },
        # "google_adk.google.adk.models.gemini_context_cache_manager": {
        #     "level": "DEBUG",
        # },
    },
    "root": {
        "handlers": ["console"],
        "level": logging.WARNING,
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
