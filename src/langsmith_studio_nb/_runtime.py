"""Effectful dependencies, gathered in one place so tests can substitute them."""

from __future__ import annotations

import atexit
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping

from langsmith_studio_nb._logging import TUNNEL_LOGGER, silence_loggers
from langsmith_studio_nb._ports import LOOPBACK
from langsmith_studio_nb._render import link_html, link_text

_TUNNEL_URL = re.compile(r"(https://[A-Za-z0-9.-]+\.trycloudflare\.com)")

tunnel_logger = logging.getLogger(TUNNEL_LOGGER)


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
    """The exact cloudflared process this session opened."""

    def kill(self) -> None:
        """Terminate the process."""
        ...


@dataclass(frozen=True)
class OpenedTunnel:
    """A Cloudflare tunnel by its public URL, owned process, and health check."""

    url: str
    process: TunnelProcess
    ready_url: str


def default_run_server(**kwargs: Any) -> None:  # noqa: ANN401 - forwards run_server's own options
    """Run the LangGraph agent server in the calling thread until it exits."""
    from langgraph_api.cli import run_server  # noqa: PLC0415 - keep the server out of import time

    run_server(**kwargs)


def capture_uvicorn_server(sink: Callable[[Any], None]) -> Callable[[], None]:
    """Hand `sink` the next Uvicorn server this process builds. Returns an undo.

    `run_server` owns `uvicorn.run`, which builds the server and blocks, so the
    only handle on it is the one taken as it is built. Do not go looking for the
    object instead. A search of the kernel finds servers this session does not
    own, and one over `sys._current_frames()` races the frames it walks.
    """
    import uvicorn  # noqa: PLC0415 - imported by the server, not by this package

    original = uvicorn.Server.__init__

    def capture(server: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401 - forwards uvicorn's own arguments
        original(server, *args, **kwargs)
        uvicorn.Server.__init__ = original  # one server per session; take the first
        sink(server)

    uvicorn.Server.__init__ = capture

    def undo() -> None:
        if uvicorn.Server.__init__ is capture:
            uvicorn.Server.__init__ = original

    return undo


class _ThreadWorker:
    """A worker that stops the one Uvicorn server it started."""

    def __init__(self, target: Callable[[], None]) -> None:
        self._server: Any = None
        self._stopping = False
        self._undo_capture = capture_uvicorn_server(self._captured)
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def _captured(self, server: Any) -> None:  # noqa: ANN401 - uvicorn's server object
        self._server = server
        if self._stopping:
            # The caller asked to stop before the server existed, so it exits as it starts.
            server.should_exit = True

    def is_alive(self) -> bool:
        """Report whether the worker thread is running."""
        return self._thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the worker thread to finish."""
        self._thread.join(timeout)
        if not self._thread.is_alive():
            # The thread ended without building a server, so none is coming.
            self._undo_capture()

    def stop(self) -> None:
        """Ask this worker's Uvicorn server to exit.

        Leaves the capture in place when the server is not up yet. `_captured`
        stops it on arrival, so a stop cannot slip past a starting server.
        """
        self._stopping = True
        if self._server is not None:
            self._server.should_exit = True


def default_spawn(target: Callable[[], None]) -> Worker:
    """Run `target` on an owned daemon thread so the notebook cell returns."""
    return _ThreadWorker(target)


def _publish_tunnel_url(stream: Any, url: Future[str]) -> None:  # noqa: ANN401 - a text pipe
    """Log everything cloudflared says, and take the tunnel URL from it."""
    for line in stream:
        text = line.rstrip()
        tunnel_logger.info("[cloudflared] %s", text)
        found = _TUNNEL_URL.search(text)
        if found and not url.done():
            url.set_result(found.group(1))


def default_open_tunnel(
    port: int, *, protocol: str | None = None, timeout: float = 30.0
) -> OpenedTunnel:
    """Open a Cloudflare quick tunnel to `port` and retain its exact process.

    Started here rather than through `langgraph_api`, which exposes no options,
    to ask cloudflared for a metrics address. Its `/ready` reports whether the
    tunnel holds a connection to Cloudflare, which neither the process nor the
    hostname reveals: cloudflared retries a blocked edge forever, and the
    hostname it printed answers for nobody in the meantime.
    """
    from langgraph_api.tunneling.cloudflare import (  # noqa: PLC0415 - optional side effect
        ensure_cloudflared,
    )

    metrics_port = default_find_free_port()
    command = [
        str(ensure_cloudflared()),
        "tunnel",
        "--url",
        f"http://{LOOPBACK}:{port}",
        "--metrics",
        f"{LOOPBACK}:{metrics_port}",
    ]
    if protocol is not None:
        command += ["--protocol", protocol]

    process = subprocess.Popen(  # noqa: S603 - fixed arguments, binary from langgraph_api
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    atexit.register(process.kill)
    url: Future[str] = Future()
    threading.Thread(target=_publish_tunnel_url, args=(process.stdout, url), daemon=True).start()
    try:
        return OpenedTunnel(
            url=url.result(timeout=timeout),
            process=process,
            ready_url=f"http://{LOOPBACK}:{metrics_port}/ready",
        )
    except BaseException:
        process.kill()
        raise


def default_status(url: str, *, timeout: float = 5.0) -> int | None:
    """Return the status `url` answers with, or None when nothing answered.

    The difference matters for a tunnel. An error from Cloudflare means it
    routes to nothing, while no answer at all means only that this host could
    not reach it.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - URL comes from the server we started
            return int(response.status)
    except urllib.error.HTTPError as error:  # a status, not a failure to reach
        return int(error.code)
    except OSError:
        return None


def default_port_is_free(port: int, *, host: str = LOOPBACK) -> bool:
    """Report whether `port` can be bound on `host`."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Matches the server's own check. Without it, a port still in TIME_WAIT
        # from the server we just stopped reads as busy.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
        return True


def default_find_free_port(*, host: str = LOOPBACK) -> int:
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


def default_in_colab() -> bool:
    """Report whether this kernel is a Colab runtime.

    Colab imports `google.colab` into every kernel it starts, and nothing else
    can import it at all.
    """
    return "google.colab" in sys.modules


def default_display_html(html: str) -> bool:
    """Render `html` in the notebook output, and report whether IPython was there to show it."""
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
    open_tunnel: Callable[..., OpenedTunnel] = default_open_tunnel
    status: Callable[[str], int | None] = default_status
    port_is_free: Callable[[int], bool] = default_port_is_free
    find_free_port: Callable[[], int] = default_find_free_port
    workspace_id: Callable[[], str | None] = default_workspace_id
    namespace: Callable[[], MutableMapping[str, Any]] = default_namespace
    in_colab: Callable[[], bool] = default_in_colab
    render: Callable[[str, str | None], None] = default_render
    quiet: Callable[[], Callable[[], None]] = default_quiet
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.monotonic
    environ: MutableMapping[str, str] = field(default_factory=lambda: os.environ)
