import contextvars
import logging
import logging.config
import sys
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
    """Inject ids into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        record.conversation_id = conversation_id_ctx.get()
        return True


def setup_logging(level: str = "INFO") -> None:
    """Single console setup for app + uvicorn; silent in frozen exe."""
    if getattr(sys, "frozen", False):
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
                "pretty": {
                    "format": "%(asctime)s | %(levelname)-5s | %(name)s [%(request_id)s/%(conversation_id)s] | %(message)s",
                    "datefmt": "%H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "pretty",
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
