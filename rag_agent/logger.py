import logging
import sys

LOGGER_NAME = "rag_agent"
_logger: logging.Logger | None = None
_verbose: bool = False

SECTION_SEP = "=" * 60
SUB_SEP = "-" * 60


def is_verbose() -> bool:
    return _verbose


def setup(verbose: bool = False) -> None:
    global _logger, _verbose
    _verbose = verbose
    _logger = logging.getLogger(LOGGER_NAME)
    _logger.handlers.clear()
    _logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    _logger.propagate = False

    if verbose:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(message)s"))
        _logger.addHandler(handler)


def get_logger() -> logging.Logger:
    if _logger is None:
        setup(verbose=False)
    return _logger


def section(title: str) -> None:
    if not _verbose:
        return
    logger = get_logger()
    logger.debug(SECTION_SEP)
    logger.debug(f"  {title}")
    logger.debug(SECTION_SEP)


def sub(title: str) -> None:
    if not _verbose:
        return
    logger = get_logger()
    logger.debug(f"\n{SUB_SEP}")
    logger.debug(f"  [{title}]")
    logger.debug(SUB_SEP)


def info(msg: str) -> None:
    if not _verbose:
        return
    get_logger().debug(f"  {msg}")


def keyval(key: str, value: str, indent: int = 4) -> None:
    if not _verbose:
        return
    prefix = " " * indent
    get_logger().debug(f"{prefix}{key}: {value}")


def content_block(title: str, content: str, max_len: int = 500) -> None:
    if not _verbose:
        return
    logger = get_logger()
    logger.debug(f"  [{title}]")
    display = content if len(content) <= max_len else content[:max_len] + "..."
    for line in display.split("\n"):
        logger.debug(f"    | {line}")
    logger.debug("")
