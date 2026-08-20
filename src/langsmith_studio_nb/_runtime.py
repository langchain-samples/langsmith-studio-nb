"""Effectful dependencies, gathered in one place so tests can substitute them."""

from __future__ import annotations

import gc
import logging
import os
import socket
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Container, Iterable, MutableMapping

from langsmith_studio_nb._logging import silence_loggers
from langsmith_studio_nb._render import link_html, link_text


class Worker(Protocol):
    """A background task running the agent server."""

    def is_alive(self) -> bool:
        """Report whether the task is still running."""
        ...

    def join(self, timeout: float | None = None) -> None:
        """Wait for the task to finish."""
        ...


def default_run_server(**kwargs: Any) -> None:  # noqa: ANN401 - forwards run_server's own options
    """Run the LangGraph agent server in the calling thread until it exits."""
    from langgraph_api.cli import run_server  # noqa: PLC0415 - keep the server out of import time

    run_server(**kwargs)


def default_spawn(target: Callable[[], None]) -> Worker:
    """Run `target` on a daemon thread so the notebook cell returns."""
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def default_probe(url: str, *, timeout: float = 5.0) -> bool:
    """Report whether `url` answers with 200."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - URL comes from the server we started
            return bool(response.status == HTTPStatus.OK)
    except OSError:
        return False


def default_port_is_free(port: int, *, host: str = "127.0.0.1") -> bool:
    """Report whether `port` can be bound on `host`."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Matches the server's own check: without it a port still in TIME_WAIT
        # from the server we just stopped reads as busy.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
        return True


def default_find_free_port(*, host: str = "127.0.0.1") -> int:
    """Return a port nothing is listening on."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def default_workspace_id() -> str | None:
    """Return the caller's LangSmith workspace, or None if it cannot be determined."""
    try:
        from langsmith import Client  # noqa: PLC0415 - optional at runtime
    except ImportError:
        return None
    try:
        response = Client().request_with_retries("GET", "/api/v1/sessions", params={"limit": 1})
        sessions = response.json()
    except Exception:  # preselecting a workspace is best effort
        return None
    return sessions[0]["tenant_id"] if sessions else None


def default_namespace() -> MutableMapping[str, Any]:
    """Return the notebook's own namespace, where cell-defined agents live."""
    return sys.modules["__main__"].__dict__


def default_modules() -> Container[str]:
    """Return the names of currently imported modules."""
    return tuple(sys.modules)


def default_live_objects() -> Iterable[Any]:
    """Return every object alive in the kernel."""
    return gc.get_objects()


def default_display_html(html: str) -> bool:
    """Render `html` in the notebook output, reporting whether it was shown."""
    try:
        from IPython.display import HTML, display  # noqa: PLC0415 - only needed inside a notebook
    except ImportError:
        return False
    display(HTML(html))
    return True


def default_render(url: str, hint: str | None = None) -> None:
    """Show the Studio link, falling back to plain text outside a notebook."""
    if not default_display_html(link_html(url, hint=hint)):
        print(link_text(url, hint=hint))


def default_quiet() -> None:
    """Silence the agent server's logs."""
    silence_loggers(logging.ERROR)


@dataclass(frozen=True)
class Runtime:
    """Every side effect `start_studio` needs, injectable for tests."""

    run_server: Callable[..., None] = default_run_server
    spawn: Callable[[Callable[[], None]], Worker] = default_spawn
    probe: Callable[[str], bool] = default_probe
    port_is_free: Callable[[int], bool] = default_port_is_free
    find_free_port: Callable[[], int] = default_find_free_port
    workspace_id: Callable[[], str | None] = default_workspace_id
    namespace: Callable[[], MutableMapping[str, Any]] = default_namespace
    modules: Callable[[], Container[str]] = default_modules
    live_objects: Callable[[], Iterable[Any]] = default_live_objects
    render: Callable[[str, str | None], None] = default_render
    quiet: Callable[[], None] = default_quiet
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.monotonic
    environ: MutableMapping[str, str] = field(default_factory=lambda: os.environ)
