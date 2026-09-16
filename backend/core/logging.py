import contextvars
import logging
import logging.config
from typing import Optional

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
conversation_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("conversation_id", default="-")


def get_request_id() -> str:
    """Current request id from context."""
    return request_id_ctx.get()


def get_conversation_id() -> str:
    """Current conversation id from context."""
    return conversation_id_ctx.get()


def set_request_id(value: Optional[object]) -> None:
    """Bind request id for this context."""
    request_id_ctx.set(str(value) if value is not None else "-")


def set_conversation_id(value: Optional[object]) -> None:
    """Bind conversation id for this context."""
    conversation_id_ctx.set(str(value) if value is not None else "-")


def clear_context() -> None:
    """Reset both ids to default."""
    request_id_ctx.set("-")
    conversation_id_ctx.set("-")


class ContextFilter(logging.Filter):
    """Inject ids and normalize uvicorn names."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        record.conversation_id = conversation_id_ctx.get()
        if record.name.startswith("uvicorn."):
            record.name = "uvicorn"
        return True


class ColoredFormatter(logging.Formatter):
    GREEN = "\x1b[32m"
    BLUE = "\x1b[34m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    LOG_FORMAT = "%(asctime)s - [%(name)s: %(levelname)s] - %(message)s"

    FORMATS = {
        logging.DEBUG: BLUE + LOG_FORMAT + RESET,
        logging.INFO: GREEN + LOG_FORMAT + RESET,
        logging.WARNING: YELLOW + LOG_FORMAT + RESET,
        logging.ERROR: RED + LOG_FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + LOG_FORMAT + RESET,
    }

    def format(self, record: logging.LogRecord) -> str:
        fmt = self.FORMATS.get(record.levelno, self.LOG_FORMAT)
        rid = getattr(record, "request_id", "-")
        cid = getattr(record, "conversation_id", "-")
        suffix = ""
        if rid != "-":
            suffix += f" [req={rid}]"
        if cid != "-":
            suffix += f" [conv={cid}]"
        if suffix:
            fmt = fmt + suffix
        formatter = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logging(level: str = "INFO") -> None:
    """Single console setup for app + uvicorn; silent in frozen exe."""
    from config import config

    if not config.IS_DEV:
        logging.disable(logging.CRITICAL)
        return


    normalized = (level or "INFO").upper()
    if normalized not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        normalized = "INFO"
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {"ctx": {"()": ContextFilter}},
            "formatters": {
                "colored": {"()": ColoredFormatter},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "colored",
                    "filters": ["ctx"],
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"handlers": ["console"], "level": normalized},
            "loggers": {
                "uvicorn": {"handlers": ["console"], "level": normalized, "propagate": False},
                "uvicorn.error": {"handlers": ["console"], "level": normalized, "propagate": False},
                "uvicorn.access": {"handlers": ["console"], "level": normalized, "propagate": False},
            },
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Return logger with context filter attached."""
    logger = logging.getLogger(name)
    if not any(isinstance(f, ContextFilter) for f in logger.filters):
        logger.addFilter(ContextFilter())
    return logger
