"""Consistent human-readable and structured logging for project entry points."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import IO, Any

from nearfield360.config import LoggingConfig

LOGGER_NAME = "nearfield360"
_MANAGED_HANDLER_ATTRIBUTE = "_nearfield360_managed"
_STANDARD_RECORD_ATTRIBUTES = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys())
_STRUCTURED_FIELDS = frozenset({"timestamp", "level", "logger", "message", "exception"})


class JsonFormatter(logging.Formatter):
    """Render one compact JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=UTC).isoformat(
            timespec="milliseconds"
        )
        payload: dict[str, Any] = {
            "timestamp": timestamp.replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {
                key: value
                for key, value in vars(record).items()
                if key not in _STANDARD_RECORD_ATTRIBUTES
                and key not in _STRUCTURED_FIELDS
                and not key.startswith("_")
            }
        )
        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"), ensure_ascii=False)


def configure_logging(config: LoggingConfig, stream: IO[str] | None = None) -> logging.Logger:
    """Configure and return the package logger without altering unrelated loggers."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in tuple(logger.handlers):
        if getattr(handler, _MANAGED_HANDLER_ATTRIBUTE, False):
            logger.removeHandler(handler)
            handler.close()

    handler = logging.StreamHandler(stream)
    setattr(handler, _MANAGED_HANDLER_ATTRIBUTE, True)
    if config.structured:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )

    level = logging.getLevelNamesMapping()[config.level]
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(component: str | None = None) -> logging.Logger:
    """Return the project logger or a namespaced child logger."""
    return logging.getLogger(LOGGER_NAME if component is None else f"{LOGGER_NAME}.{component}")


__all__ = ["LOGGER_NAME", "JsonFormatter", "configure_logging", "get_logger"]
