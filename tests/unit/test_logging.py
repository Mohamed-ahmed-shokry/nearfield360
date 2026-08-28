import io
import json
import logging

from nearfield360.config import LoggingConfig
from nearfield360.logging import LOGGER_NAME, configure_logging, get_logger


def _managed_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [
        handler for handler in logger.handlers if getattr(handler, "_nearfield360_managed", False)
    ]


def test_configure_text_logging_is_idempotent() -> None:
    first_stream = io.StringIO()
    second_stream = io.StringIO()

    logger = configure_logging(LoggingConfig(level="INFO"), stream=first_stream)
    logger = configure_logging(LoggingConfig(level="INFO"), stream=second_stream)
    get_logger("test").info("configured")

    assert first_stream.getvalue() == ""
    assert "INFO | nearfield360.test | configured" in second_stream.getvalue()
    assert len(_managed_handlers(logger)) == 1


def test_structured_logging_includes_context_and_exception() -> None:
    stream = io.StringIO()
    configure_logging(LoggingConfig(level="DEBUG", structured=True), stream=stream)

    try:
        raise RuntimeError("camera unavailable")
    except RuntimeError:
        get_logger("capture").exception(
            "capture failed",
            extra={"camera_id": "front", "frame_index": 12, "level": "spoofed"},
        )

    payload = json.loads(stream.getvalue())
    assert payload["timestamp"].endswith("Z")
    assert payload["level"] == "ERROR"
    assert payload["logger"] == "nearfield360.capture"
    assert payload["message"] == "capture failed"
    assert payload["camera_id"] == "front"
    assert payload["frame_index"] == 12
    assert "RuntimeError: camera unavailable" in payload["exception"]


def test_log_level_filters_lower_priority_records() -> None:
    stream = io.StringIO()
    configure_logging(LoggingConfig(level="WARNING"), stream=stream)

    logger = get_logger()
    logger.info("hidden")
    logger.warning("visible")

    assert "hidden" not in stream.getvalue()
    assert "visible" in stream.getvalue()
    assert logger.name == LOGGER_NAME
