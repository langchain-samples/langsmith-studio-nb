"""Effectful dependencies, gathered in one place so tests can substitute them."""

from __future__ import annotations

import atexit
import contextlib
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
from collections import deque
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
    """A background task that runs the agent server once started.

    Built and started separately, so its owner can be recorded while nothing is
    running yet. A worker that starts before anything holds it is one a failure
    on the next line leaves running with nobody to stop it.
    """

    def start(self) -> None:
        """Begin running. Call once, and only once the caller owns this."""
        ...

    def is_alive(self) -> bool:
        """Report whether the task is still running."""
        ...

    def join(self, timeout: float | None = None) -> None:
        """Wait for the task to finish."""
        ...

    def stop(self) -> None:
        """Ask the task's server to stop."""
        ...


@dataclass(frozen=True)
class OpenedTunnel:
    """A running Cloudflare tunnel, and the only handle on it."""

    url: str
    port: int  # what it forwards to; it cannot follow a server that moves
    ready_url: str  # cloudflared's own health check, which needs no DNS to answer
    close: Callable[[], None]
    """Stop the tunnel and release everything holding it.

    Must be safe to call more than once. Ownership hands a tunnel to its next
    owner before the last one lets go, which can close it twice; stranding a
    cloudflared is the failure worth avoiding, and closing one twice is not.
    """


class OpenTunnel(Protocol):
    """Opens a tunnel from the public internet to a local port."""

    def __call__(self, port: int, *, protocol: str | None = None) -> OpenedTunnel:
        """Open a tunnel to `port`, optionally over a named cloudflared protocol."""
        ...


def default_run_server(**kwargs: Any) -> None:  # noqa: ANN401 - forwards run_server's own options
    """Run the LangGraph agent server in the calling thread until it exits."""
    from langgraph_api.cli import run_server  # noqa: PLC0415 - keep the server out of import time

    run_server(**kwargs)


_capture_lock = threading.Lock()
_capture_sinks: dict[threading.Thread, Callable[[Any], None]] = {}
_capture_installed: tuple[Any, Callable[..., None]] | None = None


def _install_capture_locked(server_class: Any) -> None:  # noqa: ANN401 - uvicorn.Server
    """Route every Uvicorn server built anywhere in this process past the registry.

    One dispatcher, installed once. Do not wrap per caller: stacked wrappers
    can only be unwound in the order they went on, and a worker that outlives
    its join is never unwound at all, so the stack would grow for the life of
    the kernel.
    """
    global _capture_installed  # noqa: PLW0603 - one hook per process

    if _capture_installed is not None:
        return
    original = server_class.__init__

    def dispatch(server: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401 - forwards uvicorn's own arguments
        original(server, *args, **kwargs)
        with _capture_lock:
            sink = _forget_locked(threading.current_thread())
        if sink is not None:  # outside the lock: the sink runs caller code
            sink(server)

    _capture_installed = (server_class, original)
    server_class.__init__ = dispatch


def _forget_locked(thread: threading.Thread) -> Callable[[Any], None] | None:
    """Take `thread`'s sink, and put Uvicorn back once nobody is waiting on it."""
    global _capture_installed  # noqa: PLW0603 - one hook per process

    sink = _capture_sinks.pop(thread, None)
    if _capture_sinks or _capture_installed is None:
        return sink
    server_class, original = _capture_installed
    server_class.__init__ = original
    _capture_installed = None
    return sink


def capture_uvicorn_server(
    sink: Callable[[Any], None], *, target: Callable[[], None]
) -> threading.Thread:
    """Return an unstarted thread for `target`, and hand `sink` the server it builds.

    `run_server` owns `uvicorn.run`, which builds the server and blocks, so the
    only handle on it is the one taken as it is built. Do not go looking for the
    object instead. A search of the kernel finds servers this session does not
    own, and one over `sys._current_frames()` races the frames it walks.

    Builds the thread rather than taking one, and hands back no way to
    deregister, so a registration lasts exactly as long as its thread runs and
    only that thread ever gives it up. Nothing outside can tell the difference
    between a thread that never began and one that began and has not yet said
    so — `Thread` publishes that from inside itself — and a caller that guessed
    wrong either way would deregister a running server nothing can then stop.
    A registration for a thread that never starts is kept instead: it can fire
    for nobody, and it costs one dispatch on each Uvicorn server built after it.
    """
    import uvicorn  # noqa: PLC0415 - imported by the server, not by this package

    def run() -> None:
        try:
            target()
        finally:
            with _capture_lock:
                _forget_locked(thread)

    thread = threading.Thread(target=run, daemon=True)
    with _capture_lock:
        _capture_sinks[thread] = sink
        _install_capture_locked(uvicorn.Server)

    return thread


class _ThreadWorker:
    """A worker that stops the one Uvicorn server it started."""

    def __init__(self, target: Callable[[], None]) -> None:
        self._server: Any = None
        self._stopping = False
        self._thread = capture_uvicorn_server(self._captured, target=target)

    def start(self) -> None:
        """Run the target on this worker's thread."""
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
        with contextlib.suppress(RuntimeError):  # never started; nothing to wait for
            self._thread.join(timeout)

    def stop(self) -> None:
        """Ask this worker's Uvicorn server to exit.

        Leaves the registration in place when the server is not up yet.
        `_captured` stops it on arrival, so a stop cannot slip past a starting
        server.
        """
        self._stopping = True
        if self._server is not None:
            self._server.should_exit = True


def default_spawn(target: Callable[[], None]) -> Worker:
    """Build a worker that will run `target` on a daemon thread, so the cell returns.

    Returns it unstarted. `Worker.start` is the second phase.
    """
    return _ThreadWorker(target)


def _publish_tunnel_url(process: Any, url: Future[str], tail: deque[str]) -> None:  # noqa: ANN401 - a Popen
    """Log everything cloudflared says, and take the tunnel URL from it.

    Settles `url` on every path. Leaving it unsettled would strand the caller
    on its timeout and have it report a reader that crashed as a slow
    Cloudflare.
    """
    try:
        for line in process.stdout:
            text = line.rstrip()
            tunnel_logger.info("[cloudflared] %s", text)
            if text:
                tail.append(text)
            found = _TUNNEL_URL.search(text)
            if found and not url.done():
                url.set_result(found.group(1))
    except Exception as error:
        if not url.done():
            url.set_exception(error)
        return
    if not url.done():
        # End of the pipe means the process is on its way out, so this waits on
        # something that is already leaving.
        url.set_exception(
            RuntimeError(f"cloudflared exited with status {process.wait()}. {_last(tail)}".strip())
        )


def _last(tail: deque[str]) -> str:
    """Return the most recent line cloudflared printed, or nothing."""
    return tail[-1] if tail else ""


def default_open_tunnel(
    port: int, *, protocol: str | None = None, timeout: float = 30.0
) -> OpenedTunnel:
    """Open a Cloudflare quick tunnel to `port` and retain the only handle on it.

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

    # The exit hook goes on before the child exists, so the window in which an
    # interrupt could orphan cloudflared is the one store below rather than a
    # call into `atexit`. It cannot be closed: the child is running by the time
    # `Popen` returns, and nothing can hold it until that value is stored.
    spawned: list[Any] = []

    def kill_spawned() -> None:
        for process in spawned:
            process.kill()

    registered = atexit.register(kill_spawned)
    try:
        spawned.append(
            subprocess.Popen(  # noqa: S603 - fixed arguments, binary from langgraph_api
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
        )
    except BaseException:
        atexit.unregister(registered)
        raise
    process = spawned[0]

    def close() -> None:
        """Stop cloudflared and reap it. Safe to call more than once.

        Gives up the exit hook only once the process is really gone, so an
        interrupt anywhere in here leaves something still holding a cloudflared
        that is still running.
        """
        try:
            process.kill()
        finally:
            process.wait()  # a long session collects no zombies
        atexit.unregister(registered)

    tail: deque[str] = deque(maxlen=1)
    try:
        url: Future[str] = Future()
        threading.Thread(target=_publish_tunnel_url, args=(process, url, tail), daemon=True).start()
        opened = url.result(timeout=timeout)
    except TimeoutError as error:
        close()
        message = f"cloudflared printed no tunnel URL in {timeout:g}s. {_last(tail)}"
        raise TimeoutError(message.strip()) from error
    except BaseException:
        close()
        raise
    return OpenedTunnel(
        url=opened,
        port=port,
        ready_url=f"http://{LOOPBACK}:{metrics_port}/ready",
        close=close,
    )


def default_status(url: str, *, timeout: float = 5.0) -> int | None:
    """Return the status `url` answers with, or None when nothing answered.

    The difference matters for a tunnel. An error from Cloudflare means it
    routes to nothing, while no answer at all means only that this host could
    not reach it. Every local failure reads as no answer on purpose: this is a
    health probe in a retry loop, and it must not be the thing that raises.
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
    open_tunnel: OpenTunnel = default_open_tunnel
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
