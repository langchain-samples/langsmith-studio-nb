"""Silence the agent server's logs.

A notebook cell stops scrolling once it finishes executing, so several hundred
startup lines bury the one link the reader needs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class Leveled(Protocol):
    """Anything with a settable logging level."""

    level: int

    def setLevel(self, level: int) -> None:  # noqa: N802 - matches logging.Logger
        """Set the threshold for this logger."""
        ...


def _restore_root(level: int, handlers: list[logging.Handler]) -> None:
    """Undo what importing the server did to the root logger.

    `langgraph_api` calls `logging.basicConfig(INFO)` as it imports, which lowers
    the root level and attaches a handler to a notebook that asked for neither.
    """
    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers[:]:
        if handler not in handlers:
            root.removeHandler(handler)


TUNNEL_LOGGER = "langsmith_studio_nb.tunnel"

NOISY_LOGGERS = (
    "blockbuster",
    "httpx",
    "langgraph_api",
    "langgraph_api.server",
    "langgraph_runtime",
    "langgraph_runtime_inmem",
    "uvicorn",
    "uvicorn.error",
    TUNNEL_LOGGER,
)


def silence_loggers(
    level: int = logging.ERROR,
    *,
    names: Iterable[str] = NOISY_LOGGERS,
    get_logger: Callable[[str], Leveled] = logging.getLogger,
) -> Callable[[], None]:
    """Raise the level of the server's loggers and return a restorer.

    Safe to call before anything imports the server. `getLogger` creates a
    placeholder whose level survives the real module's import, and `langgraph_api`
    calls `logging.basicConfig(INFO)` at import time.
    """
    root = logging.getLogger()
    root_level, root_handlers = root.level, root.handlers[:]
    loggers = [get_logger(name) for name in names]
    previous = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(level)

    def restore() -> None:
        """Put every level, and the root logger, back the way it was."""
        for logger, previous_level in zip(loggers, previous, strict=True):
            logger.setLevel(previous_level)
        _restore_root(root_level, root_handlers)

    return restore
