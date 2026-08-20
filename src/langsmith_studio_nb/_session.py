"""Start and stop the agent server that LangGraph Studio connects to."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

from langsmith_studio_nb._environment import detect_environment, needs_tunnel
from langsmith_studio_nb._ports import resolve_port
from langsmith_studio_nb._runtime import Runtime, TunnelProcess, Worker
from langsmith_studio_nb._urls import studio_url

DEFAULT_GRAPH_NAME = "agent"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 2024
DEFAULT_TIMEOUT = 180.0
TUNNEL_HINT = (
    "Blocked domain? Add *.trycloudflare.com under Advanced Settings \u2192 Allowed Domains."
)
TUNNEL_FAILED = (
    "The tunnel never came up, so the server is reachable only from this kernel and "
    "Studio cannot connect to it. Cloudflare rate limits quick tunnels per IP address, "
    "which notebook hosts share, so this usually clears on its own: wait a minute and "
    "re-run this cell. Pass verbose=True to see what cloudflared reported."
)

_POLL_INTERVAL = 0.5
_JOIN_TIMEOUT = 20.0


class _Tunnel(NamedTuple):
    """An owned cloudflared process and the endpoint it forwards to."""

    url: str
    port: int
    requested: int  # what the caller asked for, so asking for another port opens another tunnel
    process: TunnelProcess


@dataclass(frozen=True)
class _SessionState:
    """Every resource owned by one running Studio session."""

    worker: Worker
    tunnel: _Tunnel | None
    restore_logging: Callable[[], None]


_state: _SessionState | None = None


@dataclass(frozen=True)
class StudioSession:
    """A running agent server and the Studio URL that opens it."""

    api_url: str
    studio_url: str
    tunnel: bool
    graphs: tuple[str, ...]

    def _repr_html_(self) -> str:
        """Render as nothing.

        `start_studio` already displayed the link; without this a bare call
        echoes this dataclass underneath it.
        """
        return ""


def stop_studio(*, runtime: Runtime | None = None) -> None:
    """Stop a running agent server and its tunnel. Safe to call when none is running."""
    _ = runtime  # retained for API compatibility with injected test runtimes
    _stop(keep_tunnel=False)


def _stop(*, keep_tunnel: bool) -> _Tunnel | None:
    """Stop owned resources, optionally returning the tunnel kept for a restart."""
    global _state  # noqa: PLW0603 - one owned session per kernel

    if _state is None:
        return None

    state = _state
    state.worker.stop()
    state.worker.join(timeout=_JOIN_TIMEOUT)
    if state.worker.is_alive():
        message = (
            f"The previous agent server did not stop within {_JOIN_TIMEOUT:g}s, "
            "so a replacement was not started."
        )
        raise RuntimeError(message)

    kept = state.tunnel if keep_tunnel else None
    try:
        if state.tunnel is not None and not keep_tunnel:
            state.tunnel.process.kill()
    finally:
        state.restore_logging()
        _state = None
    return kept


def _reusable_tunnel(*, requested: int, runtime: Runtime) -> _Tunnel | None:
    """Return the tunnel a restart can keep, if it still answers for `requested`.

    Ask the tunnel itself, while the server it forwards to is still up. A live
    `cloudflared` process proves nothing: Cloudflare drops a quick tunnel on its
    own, and a reconnect comes back under a new hostname, leaving the process
    running behind a URL that answers nothing.
    """
    if _state is None or _state.tunnel is None or _state.tunnel.requested != requested:
        return None
    return _state.tunnel if runtime.probe(f"{_state.tunnel.url}/ok") else None


def _spawn_server(
    *,
    graphs: tuple[str, ...],
    port: int,
    verbose: bool,
    runtime: Runtime,
) -> tuple[Worker, Callable[[], None]]:
    """Spawn one local server and return it with its logging restorer."""
    restore_logging = runtime.quiet() if not verbose else lambda: None
    try:
        worker = runtime.spawn(
            lambda: runtime.run_server(
                host=DEFAULT_HOST,
                port=port,
                graphs={name: f"__main__:{name}" for name in graphs},
                tunnel=False,
                reload=False,
                open_browser=False,
                allow_blocking=True,
                server_level="INFO" if verbose else "ERROR",
            )
        )
    except BaseException:
        restore_logging()
        raise
    return worker, restore_logging


def start_studio(
    *variables: str,
    port: int = DEFAULT_PORT,
    tunnel: bool | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    verbose: bool = False,
    runtime: Runtime | None = None,
) -> StudioSession:
    """Serve the agents named in `variables` and show a link that opens them in Studio.

    Pass several names to serve them side by side, and pick between them in
    Studio's graph menu. Restarts any server this kernel already started, so
    re-running the cell picks up an edited agent, and keeps the tunnel it is
    already running, so the link stays the same.

    Args:
        variables: Names of the compiled graphs in the notebook namespace.
            Defaults to `agent`.
        port: Port to serve on. A busy port is replaced with a free one.
        tunnel: Force a public tunnel on or off. Defaults to whether the
            notebook host runs the kernel away from the browser.
        timeout: Seconds to wait for the server to answer before giving up.
        verbose: Show the server's own logs, which a notebook cannot scroll.
        runtime: Injected side effects. Tests substitute this.

    Raises:
        NameError: No such variable in the notebook namespace.
        RuntimeError: The server stopped while starting up, or the tunnel never
            came up.
        TimeoutError: The server did not answer within `timeout`.
    """
    global _state  # noqa: PLW0603 - one owned session per kernel

    runtime = runtime or Runtime()
    graphs = variables or (DEFAULT_GRAPH_NAME,)
    namespace = runtime.namespace()
    missing = [name for name in graphs if name not in namespace]
    if missing:
        message = (
            f"No variable named {missing[0]!r} in the notebook. "
            "Run the cell that defines your agent first."
        )
        raise NameError(message)

    if tunnel is None:
        environment = detect_environment(modules=runtime.modules(), environ=runtime.environ)
        tunnel = needs_tunnel(environment)

    requested = port
    reused = _reusable_tunnel(requested=requested, runtime=runtime) if tunnel else None
    reused = _stop(keep_tunnel=reused is not None)
    try:
        # The tunnel forwards to the port it was opened on, so a restart goes back there.
        port = resolve_port(
            reused.port if reused else requested,
            is_free=runtime.port_is_free,
            find_free=runtime.find_free_port,
        )
        if reused is not None and reused.port != port:
            # Something else took the port, so the tunnel now forwards to nothing.
            reused.process.kill()
            reused = None

        worker, restore_logging = _spawn_server(
            graphs=graphs, port=port, verbose=verbose, runtime=runtime
        )
    except BaseException:
        if reused is not None:
            reused.process.kill()
        raise
    _state = _SessionState(worker=worker, tunnel=reused, restore_logging=restore_logging)
    try:
        local_api_url = f"http://{DEFAULT_HOST}:{port}"
        _wait_for_server(worker, runtime=runtime, timeout=timeout, api_url=local_api_url)
        api_url = local_api_url
        if tunnel:
            if reused is None:
                try:
                    opened = runtime.open_tunnel(port)
                except Exception as error:
                    raise RuntimeError(TUNNEL_FAILED) from error
                reused = _Tunnel(opened.url, port, requested, opened.process)
                _state = _SessionState(
                    worker=worker, tunnel=reused, restore_logging=restore_logging
                )
            api_url = reused.url

        url = studio_url(api_url, workspace_id=runtime.workspace_id())
        runtime.render(url, TUNNEL_HINT if tunnel else None)
        return StudioSession(api_url=api_url, studio_url=url, tunnel=tunnel, graphs=graphs)
    except BaseException as error:
        try:
            _stop(keep_tunnel=False)
        except RuntimeError as cleanup_error:
            error.add_note(str(cleanup_error))
        raise


def _wait_for_server(worker: Worker, *, runtime: Runtime, timeout: float, api_url: str) -> None:
    """Return once the local server answers, or raise."""
    deadline = runtime.now() + timeout
    while runtime.now() < deadline:
        if not worker.is_alive():
            message = "The agent server stopped while starting up. Check the log above."
            raise RuntimeError(message)
        if runtime.probe(f"{api_url}/ok"):
            return
        runtime.sleep(_POLL_INTERVAL)
    message = (
        f"The agent server did not answer within {timeout:g}s. "
        "Re-run this cell, or pass verbose=True to see the server's logs."
    )
    raise TimeoutError(message)
