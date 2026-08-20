"""Choose the port to serve on."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def resolve_port(port: int, *, is_free: Callable[[int], bool], find_free: Callable[[], int]) -> int:
    """Return `port`, or a free one when something already listens on it.

    The server picks its own replacement port when the requested one is busy,
    but never reports which. Choosing here keeps the port we poll for readiness
    the port the server is on; otherwise a stale server answers for it.
    """
    return port if is_free(port) else find_free()
