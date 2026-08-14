"""Effectful dependencies, gathered in one place so tests can substitute them."""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Container, Iterable, MutableMapping

from langsmith_studio_nb._render import render


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


def default_display_html(html: str) -> None:
    """Render `html` in the notebook output, when there is one."""
    try:
        from IPython.display import HTML, display  # noqa: PLC0415 - only needed inside a notebook
    except ImportError:  # no notebook front end; the plain URL still prints
        return
    display(HTML(html))


def default_render(url: str) -> None:
    """Show the Studio link in the notebook output."""
    render(url, display_html=default_display_html, echo=print)


@dataclass(frozen=True)
class Runtime:
    """Every side effect `start_studio` needs, injectable for tests."""

    run_server: Callable[..., None] = default_run_server
    spawn: Callable[[Callable[[], None]], Worker] = default_spawn
    probe: Callable[[str], bool] = default_probe
    workspace_id: Callable[[], str | None] = default_workspace_id
    namespace: Callable[[], MutableMapping[str, Any]] = default_namespace
    modules: Callable[[], Container[str]] = default_modules
    live_objects: Callable[[], Iterable[Any]] = default_live_objects
    render: Callable[[str], None] = default_render
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.monotonic
    environ: MutableMapping[str, str] = field(default_factory=lambda: os.environ)
