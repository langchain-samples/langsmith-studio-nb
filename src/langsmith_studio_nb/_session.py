"""Start and stop the agent server that LangGraph Studio connects to."""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, replace
from http import HTTPStatus
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

from langsmith_studio_nb._policy import (
    DEFAULT_PORT,
    DEFAULT_TIMEOUT,
    TUNNEL_PROTOCOLS,
    Request,
    plan,
    should_tunnel,
    tunnel_still_reaches,
)
from langsmith_studio_nb._ports import LOOPBACK, resolve_port
from langsmith_studio_nb._runtime import OpenedTunnel, Runtime, Worker
from langsmith_studio_nb._urls import port_of, studio_url

TUNNEL_HINT = "Blocked domain? Add *.trycloudflare.com under Advanced Settings → Allowed Domains."
TUNNEL_FAILED = (
    "The tunnel never came up, so the server is reachable only from this kernel and "
    "Studio cannot connect to it. Cloudflare rate limits quick tunnels per IP address, "
    "which every notebook on a Colab host shares, so this usually clears on its own. "
    "Wait a minute and re-run this cell. Pass verbose=True to see what cloudflared "
    "reported."
)
TUNNEL_ERROR = "The tunnel could not be opened: "
TUNNEL_UNREACHABLE = (
    "Opened {attempts} tunnels and cloudflared never reached Cloudflare with any of "
    "them, so their URLs answer for nobody and Studio could not have connected. A "
    "tunnel needs port 7844 out of this runtime, UDP for the default protocol and TCP "
    "for the http2 one this also tried, and some runtimes allow neither. Runtime "
    "\N{RIGHTWARDS ARROW} Disconnect and delete runtime, then reconnect, usually lands "
    "on one that does. Pass verbose=True to see what cloudflared reported."
)

_API_URL_VARIABLE = "LANGGRAPH_API_URL"
_POLL_INTERVAL = 0.5
_JOIN_TIMEOUT = 20.0
_TUNNEL_READY_TIMEOUT = 15.0
_TUNNEL_RETRY_PAUSE = 2.0


@dataclass(frozen=True)
class _SessionState:
    """Every resource owned by one running Studio session."""

    worker: Worker
    requested_port: int  # what the caller asked for, so asking for another opens another tunnel
    tunnel: OpenedTunnel | None
    restore_logging: Callable[[], None]
    restore_api_url: Callable[[], None]


# Held across each whole ownership transition rather than around the assignment
# to `_state`. Two starts that interleave would each stop the other's server and
# leave the loser's worker and tunnel running with nothing holding them.
_lock = threading.RLock()
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


def stop_studio() -> None:
    """Stop a running agent server and its tunnel. Safe to call when none is running.

    Waits for a `start_studio` already in flight rather than tearing down the
    session it is about to hand back.
    """
    with _lock:
        _stop()


def _release_all(steps: Sequence[Callable[[], None]]) -> None:
    """Run every step, whatever an earlier one does.

    Teardown has nothing left to fall back on, so a step that fails must not
    take the steps after it down with it. An interrupt still lands, but only
    once everything has been given up.
    """
    if not steps:
        return
    try:
        with contextlib.suppress(Exception):
            steps[0]()
    finally:
        _release_all(steps[1:])


def _steps_to_release(
    state: _SessionState, *, keeping: OpenedTunnel | None
) -> list[Callable[[], None]]:
    """Return everything `state` owns that `keeping` does not carry forward."""
    steps: list[Callable[[], None]] = []
    if state.tunnel is not None and state.tunnel is not keeping:
        steps.append(state.tunnel.close)
    return [*steps, state.restore_api_url, state.restore_logging]


def _stop(*, keep: list[OpenedTunnel] | None = None) -> None:
    """Stop owned resources, handing a tunnel a restart can keep to `keep`.

    Releases what it can and never raises short of an interrupt. This is the
    only path that gives up the tunnel, the environment variable, and the log
    levels, so a server that will not stop must not take them down with it.

    `keep` is the handoff rather than the return value, because a tunnel that
    survives has to reach its next owner before anything else is released. An
    interrupt during teardown would strand a tunnel this never got to return.
    """
    global _state

    if _state is None:
        return

    state, _state = _state, None
    kept: OpenedTunnel | None = None
    try:
        with contextlib.suppress(Exception):
            state.worker.stop()
            # A server draining a long request outlives the join, but gives up
            # its port as it starts shutting down, so the tunnel is still worth
            # keeping. The next server reports the port it got, which settles
            # whether it can use this one.
            state.worker.join(timeout=_JOIN_TIMEOUT)
            if keep is not None and state.tunnel is not None:
                # Hand it on first. Closing is safe twice and stranding is not,
                # so the new owner has to have it before the old one lets go.
                keep.append(state.tunnel)
                kept = state.tunnel  # only a clean stop hands one on
    finally:
        _release_all(_steps_to_release(state, keeping=kept))


def _has_reusable_tunnel(*, requested: int, runtime: Runtime) -> bool:
    """Report whether the running session's tunnel can survive this restart.

    A live `cloudflared` process proves nothing. Cloudflare drops a quick tunnel
    on its own, which leaves the process retrying behind a URL that answers
    nothing, and cloudflared is the one that knows.
    """
    if _state is None or _state.tunnel is None or _state.requested_port != requested:
        return False
    return _tunnel_is_up(_state.tunnel, runtime=runtime)


def _answers(url: str, *, runtime: Runtime) -> bool:
    """Report whether `url` serves the agent server right now."""
    return runtime.status(f"{url}/ok") == HTTPStatus.OK


def _tunnel_is_up(tunnel: OpenedTunnel, *, runtime: Runtime) -> bool:
    """Report whether cloudflared holds a connection to Cloudflare for this tunnel.

    Ask cloudflared, not the hostname. A quick tunnel URL is handed out before
    any connection exists and keeps resolving to nothing while cloudflared
    retries an edge it cannot reach, and the kernel's own view of that hostname
    is worth even less: a new one can take minutes to resolve here while the
    browser has it at once.
    """
    return runtime.status(tunnel.ready_url) == HTTPStatus.OK


def _still_serves(tunnel: OpenedTunnel, *, bound: int, runtime: Runtime) -> bool:
    """Report whether `tunnel` reaches the server on `bound` and is up right now."""
    return tunnel_still_reaches(tunnel_port=tunnel.port, bound_port=bound) and _tunnel_is_up(
        tunnel, runtime=runtime
    )


def _wait_for_tunnel(opened: OpenedTunnel, *, runtime: Runtime) -> bool:
    """Report whether the tunnel connects to Cloudflare before the deadline."""
    deadline = runtime.now() + _TUNNEL_READY_TIMEOUT
    while runtime.now() < deadline:
        if _tunnel_is_up(opened, runtime=runtime):
            return True
        runtime.sleep(_POLL_INTERVAL)
    return False


def _open_verified_tunnel(
    state: _SessionState, *, port: int, runtime: Runtime
) -> tuple[_SessionState, OpenedTunnel]:
    """Own a tunnel that is connected to Cloudflare, or raise.

    Walks `TUNNEL_PROTOCOLS` once each. Both the count and the order are a
    rate limit this host shares, spelled out there. The session takes ownership
    before this returns, so an interrupt on the way out cannot strand one.
    """
    last = len(TUNNEL_PROTOCOLS) - 1
    for attempt, protocol in enumerate(TUNNEL_PROTOCOLS):
        try:
            opened = runtime.open_tunnel(port, protocol=protocol)
        except TimeoutError as error:
            # Cloudflare withheld the URL, which is what rate limiting looks
            # like. Another attempt spends another tunnel to be told the same.
            raise RuntimeError(f"{TUNNEL_FAILED} ({error})") from error
        except Exception as error:
            raise RuntimeError(f"{TUNNEL_ERROR}{error}") from error
        try:
            connected = _wait_for_tunnel(opened, runtime=runtime)
            if connected:
                state = _own_tunnel(state, opened)  # owned before anything else can raise
        except BaseException:
            # Nothing owns this tunnel yet, so an interrupt here strands it.
            opened.close()
            raise
        if connected:
            return state, opened
        opened.close()
        if attempt < last:
            runtime.sleep(_TUNNEL_RETRY_PAUSE)
    raise RuntimeError(TUNNEL_UNREACHABLE.format(attempts=len(TUNNEL_PROTOCOLS)))


def _lease_api_url(runtime: Runtime) -> Callable[[], None]:
    """Take `LANGGRAPH_API_URL` for this session, and return a restorer.

    The server publishes the URL it bound here, so this session has to be its
    only writer. Whatever is there now belongs to someone else, an older server
    of ours or the caller's own setting, and reading it back would point this
    session at an endpoint it does not own. Putting it back on the way out
    leaves a dead URL of ours in nobody's environment.
    """
    previous = runtime.environ.pop(_API_URL_VARIABLE, None)

    def restore() -> None:
        if previous is None:
            runtime.environ.pop(_API_URL_VARIABLE, None)
        else:
            runtime.environ[_API_URL_VARIABLE] = previous

    return restore


def _spawn_server(
    request: Request, *, port: int, tunnel: OpenedTunnel | None, verbose: bool, runtime: Runtime
) -> _SessionState:
    """Start one local server, and own it before it runs.

    Two phases on purpose. `runtime.spawn` builds a worker without starting it,
    the session takes ownership, and only then does it run. Anything that fails
    from here on is something teardown can find.
    """
    restore_logging = runtime.quiet() if not verbose else lambda: None
    restore_api_url = _lease_api_url(runtime)
    try:
        worker = runtime.spawn(
            lambda: runtime.run_server(
                host=LOOPBACK,
                port=port,
                graphs={name: f"__main__:{name}" for name in request.graphs},
                tunnel=False,
                reload=False,
                open_browser=False,
                allow_blocking=True,
                server_level="INFO" if verbose else "ERROR",
            )
        )
    except BaseException:
        _release_all([restore_api_url, restore_logging])
        raise
    state = _own(
        _SessionState(
            worker=worker,
            requested_port=request.port,
            tunnel=tunnel,
            restore_logging=restore_logging,
            restore_api_url=restore_api_url,
        )
    )
    worker.start()  # owned first, so nothing runs that teardown cannot find
    return state


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
        tunnel: Force a public tunnel on or off. Defaults to on under Colab,
            whose kernel the browser cannot reach directly, and off elsewhere.
        timeout: Seconds to wait for the server to answer before giving up.
        verbose: Show the server's own logs, which a notebook cannot scroll.
        runtime: Injected side effects. Tests substitute this.

    Raises:
        NameError: No such variable in the notebook namespace.
        RuntimeError: The server stopped while starting up, or the tunnel never
            came up.
        TimeoutError: The server did not answer within `timeout`.
    """
    runtime = runtime or Runtime()
    request = plan(
        variables,
        port=port,
        tunnel=should_tunnel(tunnel, in_colab=runtime.in_colab()),
        namespace=runtime.namespace(),
    )
    with _lock:
        return _start(request, timeout=timeout, verbose=verbose, runtime=runtime)


def _own(state: _SessionState) -> _SessionState:
    """Publish what this session owns, so teardown can find it."""
    global _state  # noqa: PLW0603 - one owned session per kernel

    _state = state
    return state


def _own_tunnel(state: _SessionState, tunnel: OpenedTunnel | None) -> _SessionState:
    """Record the tunnel this session owns. Call before closing or opening, never after.

    Between the two, a failure hands teardown a tunnel that is already gone or
    misses one that is not.
    """
    return _own(replace(state, tunnel=tunnel))


def _start(request: Request, *, timeout: float, verbose: bool, runtime: Runtime) -> StudioSession:
    """Own one server, and the tunnel it needs, for the rest of this kernel."""
    keep_tunnel = request.tunnel and _has_reusable_tunnel(requested=request.port, runtime=runtime)
    handoff: list[OpenedTunnel] = []
    try:
        _stop(keep=handoff if keep_tunnel else None)
        reused = handoff[0] if handoff else None
        # The tunnel forwards to the port we opened it on, so a restart goes back there.
        port = resolve_port(
            reused.port if reused else request.port,
            is_free=runtime.port_is_free,
            find_free=runtime.find_free_port,
        )
        state = _spawn_server(request, port=port, tunnel=reused, verbose=verbose, runtime=runtime)
    except BaseException:
        _stop()  # whatever `_spawn_server` got as far as owning
        _release_all([tunnel.close for tunnel in handoff])  # and what never reached it
        raise
    try:
        api_url = _wait_for_server(state.worker, runtime=runtime, timeout=timeout)
        if request.tunnel:
            # The server resolves the port again itself and moves off a busy one
            # without saying so, so take the port it reports, not the one we asked for.
            bound = port_of(api_url) or port
            if reused is not None and not _still_serves(reused, bound=bound, runtime=runtime):
                # Checked last thing before the link goes out, because this is
                # the only check that speaks for the link. The one that kept
                # this tunnel ran before the old server stopped, and Cloudflare
                # can drop a tunnel in the time the next one takes to answer.
                state = _own_tunnel(state, None)
                reused.close()
                reused = None
            if reused is None:
                state, reused = _open_verified_tunnel(state, port=bound, runtime=runtime)
            api_url = reused.url

        url = studio_url(api_url, workspace_id=runtime.workspace_id())
        runtime.render(url, TUNNEL_HINT if request.tunnel else None)
        return StudioSession(
            api_url=api_url, studio_url=url, tunnel=request.tunnel, graphs=request.graphs
        )
    except BaseException:
        _stop()
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
