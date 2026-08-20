"""Start and stop the agent server that LangGraph Studio connects to."""

from __future__ import annotations

from dataclasses import dataclass

from langsmith_studio_nb._environment import detect_environment, needs_tunnel
from langsmith_studio_nb._ports import resolve_port
from langsmith_studio_nb._runtime import Runtime, Worker
from langsmith_studio_nb._teardown import shut_down
from langsmith_studio_nb._urls import is_loopback_url, studio_url

DEFAULT_GRAPH_NAME = "agent"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 2024
DEFAULT_TIMEOUT = 180.0
TUNNEL_HINT = (
    "Blocked domain? Add *.trycloudflare.com under Advanced Settings \u2192 Allowed Domains."
)
TUNNEL_FAILED = (
    "The tunnel never came up, so the server is reachable only from this kernel and "
    "Studio cannot connect to it. Re-run this cell, or pass verbose=True to see why "
    "cloudflared did not start."
)

_API_URL_VARIABLE = "LANGGRAPH_API_URL"
_POLL_INTERVAL = 0.5
_JOIN_TIMEOUT = 20.0

_active: Worker | None = None


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
    global _active  # noqa: PLW0603 - one server per kernel, tracked at module scope

    runtime = runtime or Runtime()
    shut_down(runtime.live_objects())
    if _active is not None:
        # run_server restores the environment as it unwinds; a restart that
        # overlaps that would have its API URL wiped out.
        _active.join(timeout=_JOIN_TIMEOUT)
        _active = None
    runtime.environ.pop(_API_URL_VARIABLE, None)


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
    re-running the cell picks up an edited agent.

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
    global _active  # noqa: PLW0603 - one server per kernel, tracked at module scope

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

    if not verbose:
        runtime.quiet()

    stop_studio(runtime=runtime)
    port = resolve_port(port, is_free=runtime.port_is_free, find_free=runtime.find_free_port)
    _active = runtime.spawn(
        lambda: runtime.run_server(
            host=DEFAULT_HOST,
            port=port,
            graphs={name: f"__main__:{name}" for name in graphs},
            tunnel=tunnel,
            reload=False,
            open_browser=False,
            allow_blocking=True,
            server_level="INFO" if verbose else "ERROR",
        )
    )

    api_url = _wait_for_server(_active, runtime=runtime, timeout=timeout, port=port, tunnel=tunnel)
    if tunnel and is_loopback_url(api_url):
        # The server falls back to its local URL when the tunnel never reports
        # one. Rendering that gives Studio a link it can only fail to fetch.
        stop_studio(runtime=runtime)
        raise RuntimeError(TUNNEL_FAILED)

    url = studio_url(api_url, workspace_id=runtime.workspace_id())
    runtime.render(url, TUNNEL_HINT if tunnel else None)
    return StudioSession(api_url=api_url, studio_url=url, tunnel=tunnel, graphs=graphs)


def readiness_url(api_url: str, *, port: int, tunnel: bool) -> str:
    """Return the URL to poll while waiting for the server.

    Never poll the tunnel: its hostname can take minutes to resolve from the
    kernel while resolving immediately in the browser, and only the browser's
    view of it matters.
    """
    return f"http://{DEFAULT_HOST}:{port}/ok" if tunnel else f"{api_url}/ok"


def _wait_for_server(
    worker: Worker, *, runtime: Runtime, timeout: float, port: int, tunnel: bool
) -> str:
    """Return the server's base URL once it answers, or raise."""
    deadline = runtime.now() + timeout
    while runtime.now() < deadline:
        if not worker.is_alive():
            message = "The agent server stopped while starting up. Check the log above."
            raise RuntimeError(message)
        api_url = runtime.environ.get(_API_URL_VARIABLE)
        if api_url and runtime.probe(readiness_url(api_url, port=port, tunnel=tunnel)):
            return api_url
        runtime.sleep(_POLL_INTERVAL)
    message = (
        f"The agent server did not answer within {timeout:g}s. "
        "Re-run this cell, or pass verbose=True to see the server's logs."
    )
    raise TimeoutError(message)
