"""Effectful dependencies, gathered in one place so tests can substitute them."""

from __future__ import annotations

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
    from collections.abc import Callable, Container, MutableMapping
    from types import FrameType

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

    def stop(self) -> None:
        """Ask the task's server to stop."""
        ...


class TunnelProcess(Protocol):
    """The exact cloudflared process opened for this session."""

    def kill(self) -> None:
        """Terminate the process."""
        ...


@dataclass(frozen=True)
class OpenedTunnel:
    """A Cloudflare tunnel by its public URL and owned process."""

    url: str
    process: TunnelProcess


def default_run_server(**kwargs: Any) -> None:  # noqa: ANN401 - forwards run_server's own options
    """Run the LangGraph agent server in the calling thread until it exits."""
    from langgraph_api.cli import run_server  # noqa: PLC0415 - keep the server out of import time

    run_server(**kwargs)


def _is_uvicorn_server(candidate: Any) -> bool:  # noqa: ANN401 - scans frame locals
    """Report whether `candidate` is Uvicorn's server object."""
    kind = type(candidate)
    return (
        kind.__name__ == "Server"
        and kind.__module__.split(".")[0] == "uvicorn"
        and hasattr(candidate, "should_exit")
    )


def _request_server_exit(frame: FrameType | None) -> None:
    """Stop the Uvicorn server owned by one worker thread, if it has started."""
    while frame is not None:
        for candidate in frame.f_locals.values():
            if _is_uvicorn_server(candidate):
                candidate.should_exit = True
                return
        frame = frame.f_back


class _ThreadWorker:
    """A worker that can stop only the Uvicorn server on its own thread."""

    def __init__(self, target: Callable[[], None]) -> None:
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def is_alive(self) -> bool:
        """Report whether the worker thread is running."""
        return self._thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the worker thread to finish."""
        self._thread.join(timeout)

    def stop(self) -> None:
        """Ask this thread's Uvicorn server to exit."""
        ident = self._thread.ident
        if ident is not None:
            _request_server_exit(sys._current_frames().get(ident))


def default_spawn(target: Callable[[], None]) -> Worker:
    """Run `target` on an owned daemon thread so the notebook cell returns."""
    return _ThreadWorker(target)


def default_open_tunnel(port: int, *, timeout: float = 30.0) -> OpenedTunnel:
    """Open a Cloudflare quick tunnel and retain its exact process."""
    from langgraph_api.tunneling.cloudflare import (  # noqa: PLC0415 - optional side effect
        start_tunnel,
    )

    tunnel = start_tunnel(port)
    try:
        url = tunnel.url.result(timeout=timeout)
    except BaseException:
        tunnel.process.kill()
        raise
    return OpenedTunnel(url=url, process=tunnel.process)


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


def default_quiet() -> Callable[[], None]:
    """Silence the agent server's logs and return a restorer."""
    return silence_loggers(logging.ERROR)


@dataclass(frozen=True)
class Runtime:
    """Every side effect `start_studio` needs, injectable for tests."""

    run_server: Callable[..., None] = default_run_server
    spawn: Callable[[Callable[[], None]], Worker] = default_spawn
    open_tunnel: Callable[[int], OpenedTunnel] = default_open_tunnel
    probe: Callable[[str], bool] = default_probe
    port_is_free: Callable[[int], bool] = default_port_is_free
    find_free_port: Callable[[], int] = default_find_free_port
    workspace_id: Callable[[], str | None] = default_workspace_id
    namespace: Callable[[], MutableMapping[str, Any]] = default_namespace
    modules: Callable[[], Container[str]] = default_modules
    render: Callable[[str, str | None], None] = default_render
    quiet: Callable[[], Callable[[], None]] = default_quiet
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.monotonic
    environ: MutableMapping[str, str] = field(default_factory=lambda: os.environ)
