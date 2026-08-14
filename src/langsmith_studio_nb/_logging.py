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
    """Anything whose logging level can be raised."""

    def setLevel(self, level: int) -> None:  # noqa: N802 - matches logging.Logger
        """Set the threshold for this logger."""
        ...


NOISY_LOGGERS = (
    "blockbuster",
    "httpx",
    "langgraph_api",
    "langgraph_runtime",
    "langgraph_runtime_inmem",
    "uvicorn",
)


def silence_loggers(
    level: int = logging.ERROR,
    *,
    names: Iterable[str] = NOISY_LOGGERS,
    get_logger: Callable[[str], Leveled] = logging.getLogger,
) -> None:
    """Raise the level of the server's loggers.

    Safe to call before the server is imported: `getLogger` creates a placeholder
    whose level survives the real module's import, and `langgraph_api` calls
    `logging.basicConfig(INFO)` at import time.
    """
    for name in names:
        get_logger(name).setLevel(level)
