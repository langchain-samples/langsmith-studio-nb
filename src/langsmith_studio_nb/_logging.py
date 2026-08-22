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
    loggers = [get_logger(name) for name in names]
    previous = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(level)

    def restore() -> None:
        """Restore every logger to the level it had before this call."""
        for logger, previous_level in zip(loggers, previous, strict=True):
            logger.setLevel(previous_level)

    return restore
