"""Start and stop the agent server that LangGraph Studio connects to."""

from __future__ import annotations

from dataclasses import dataclass, replace
from http import HTTPStatus
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

from langsmith_studio_nb._environment import detect_environment, needs_tunnel
from langsmith_studio_nb._ports import resolve_port
from langsmith_studio_nb._runtime import Runtime, TunnelProcess, Worker
from langsmith_studio_nb._urls import port_of, studio_url

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
    "which notebook hosts share, so this usually clears on its own. Wait a minute and "
    "re-run this cell. Pass verbose=True to see what cloudflared reported."
)
TUNNEL_ERROR = "The tunnel could not be opened: "
TUNNEL_UNREACHABLE = (
    "Opened {attempts} tunnels and Cloudflare could not route any of them to this "
    "server, so Studio would not have reached it either. Quick tunnels are best "
    "effort and some come up dead. Wait a minute and re-run this cell."
)

_API_URL_VARIABLE = "LANGGRAPH_API_URL"
_POLL_INTERVAL = 0.5
_JOIN_TIMEOUT = 20.0
_TUNNEL_ATTEMPTS = 3
_TUNNEL_READY_TIMEOUT = 10.0
_TUNNEL_RETRY_PAUSE = 2.0


class _Tunnel(NamedTuple):
    """An owned cloudflared process and the endpoint it forwards to."""

    url: str
    port: int
    requested: int  # what the caller asked for, so asking for another port opens another tunnel
    process: TunnelProcess
    confirmed: bool  # this host reached it once, so silence from it now means something


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
    _ = runtime  # unused, but injected test runtimes pass one
    _stop(keep_tunnel=False)


def _stop(*, keep_tunnel: bool) -> _Tunnel | None:
    """Stop owned resources, optionally returning the tunnel a restart can keep.

    Releases what it can and never raises. This is the only path that kills the
    tunnel and restores logging, so a server that will not stop must not take
    them down with it.
    """
    global _state

    if _state is None:
        return None

    state, _state = _state, None
    state.worker.stop()
    # A server draining a long request outlives the join, but gives up its port
    # as it starts shutting down, so the tunnel is still worth keeping. The next
    # server reports the port it got, which settles whether it can use this one.
    state.worker.join(timeout=_JOIN_TIMEOUT)
    kept = state.tunnel if keep_tunnel else None
    try:
        if state.tunnel is not None and kept is None:
            state.tunnel.process.kill()
    finally:
        state.restore_logging()
    return kept


def _reusable_tunnel(*, requested: int, runtime: Runtime) -> _Tunnel | None:
    """Return the tunnel a restart can keep, if it still serves `requested`.

    Ask the tunnel itself, while the server it forwards to is still up. A live
    `cloudflared` process proves nothing. Cloudflare drops a quick tunnel on its
    own, and a reconnect comes back under a new hostname, which leaves the
    process running behind a URL that answers nothing.
    """
    if _state is None or _state.tunnel is None or _state.tunnel.requested != requested:
        return None
    tunnel = _state.tunnel
    reached = _reached(tunnel.url, runtime=runtime)
    if reached is not None:
        return tunnel._replace(confirmed=True) if reached else None
    # Nothing answered. That only rules out a tunnel this host has reached
    # before. One it never reached is no worse off than when it opened.
    return None if tunnel.confirmed else tunnel


def _answers(url: str, *, runtime: Runtime) -> bool:
    """Report whether `url` serves the agent server right now."""
    return runtime.status(f"{url}/ok") == HTTPStatus.OK


def _reached(url: str, *, runtime: Runtime) -> bool | None:
    """Report what `url` says: True the server, False something else, None nothing."""
    status = runtime.status(f"{url}/ok")
    return None if status is None else status == HTTPStatus.OK


def _confirm_new_tunnel(url: str, *, runtime: Runtime) -> bool | None:
    """Wait out DNS for a verdict on a tunnel this session just opened.

    A tunnel that works answers the first time, so only a tunnel in doubt costs
    the wait. Cloudflare answers for a tunnel it cannot route, and that one is
    worth replacing, but silence is not a verdict. A new hostname can take
    minutes to resolve here while the browser reaches it straight away, and
    re-opening on that guess spends a tunnel against a rate limit the whole
    notebook host shares.
    """
    deadline = runtime.now() + _TUNNEL_READY_TIMEOUT
    verdict = None
    while runtime.now() < deadline:
        verdict = _reached(url, runtime=runtime)
        if verdict:
            return True
        runtime.sleep(_POLL_INTERVAL)
    return verdict


def _open_verified_tunnel(*, port: int, requested: int, runtime: Runtime) -> _Tunnel:
    """Open a tunnel that answers, or raise. Never returns an unreachable URL."""
    for attempt in range(1, _TUNNEL_ATTEMPTS + 1):
        try:
            opened = runtime.open_tunnel(port)
        except TimeoutError as error:
            # Cloudflare withheld the URL, which is what rate limiting looks
            # like. Another attempt spends another tunnel to be told the same.
            raise RuntimeError(TUNNEL_FAILED) from error
        except Exception as error:
            raise RuntimeError(f"{TUNNEL_ERROR}{error}") from error
        confirmed = _confirm_new_tunnel(opened.url, runtime=runtime)
        if confirmed is not False:
            return _Tunnel(opened.url, port, requested, opened.process, confirmed is True)
        opened.process.kill()
        if attempt < _TUNNEL_ATTEMPTS:
            runtime.sleep(_TUNNEL_RETRY_PAUSE)
    raise RuntimeError(TUNNEL_UNREACHABLE.format(attempts=_TUNNEL_ATTEMPTS))


def _spawn_server(
    *,
    graphs: tuple[str, ...],
    port: int,
    verbose: bool,
    runtime: Runtime,
) -> tuple[Worker, Callable[[], None]]:
    """Spawn one local server and return it with its logging restorer."""
    restore_logging = runtime.quiet() if not verbose else lambda: None
    # Whatever is there now belongs to a server that is gone, and reading it
    # back would point this session at an endpoint it does not own.
    runtime.environ.pop(_API_URL_VARIABLE, None)
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
        port: Port to serve on. If it is busy, this picks a free one instead.
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
        # The tunnel forwards to the port we opened it on, so a restart goes back there.
        port = resolve_port(
            reused.port if reused else requested,
            is_free=runtime.port_is_free,
            find_free=runtime.find_free_port,
        )
        worker, restore_logging = _spawn_server(
            graphs=graphs, port=port, verbose=verbose, runtime=runtime
        )
    except BaseException:
        if reused is not None:
            reused.process.kill()
        raise
    _state = _SessionState(worker=worker, tunnel=reused, restore_logging=restore_logging)
    try:
        api_url = _wait_for_server(worker, runtime=runtime, timeout=timeout)
        # The server resolves the port again itself and moves off a busy one
        # without saying so, so take the port it reports, not the one we asked for.
        bound = port_of(api_url) or port
        if tunnel:
            if reused is not None and reused.port != bound:
                # The server moved; the tunnel still forwards to where it was.
                reused.process.kill()
                reused = None
            if reused is None:
                reused = _open_verified_tunnel(port=bound, requested=requested, runtime=runtime)
            _state = replace(_state, tunnel=reused)
            api_url = reused.url

        url = studio_url(api_url, workspace_id=runtime.workspace_id())
        runtime.render(url, TUNNEL_HINT if tunnel else None)
        return StudioSession(api_url=api_url, studio_url=url, tunnel=tunnel, graphs=graphs)
    except BaseException:
        _stop(keep_tunnel=False)
        raise


def _wait_for_server(worker: Worker, *, runtime: Runtime, timeout: float) -> str:
    """Return the URL the server publishes for itself once it answers, or raise."""
    deadline = runtime.now() + timeout
    while runtime.now() < deadline:
        if not worker.is_alive():
            message = "The agent server stopped while starting up. Check the log above."
            raise RuntimeError(message)
        api_url = runtime.environ.get(_API_URL_VARIABLE)
        if api_url and _answers(api_url, runtime=runtime):
            return api_url
        runtime.sleep(_POLL_INTERVAL)
    message = (
        f"The agent server did not answer within {timeout:g}s. "
        "Re-run this cell, or pass verbose=True to see the server's logs."
    )
    raise TimeoutError(message)
