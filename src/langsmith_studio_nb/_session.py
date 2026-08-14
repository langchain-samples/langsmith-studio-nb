"""Start and stop the agent server that LangGraph Studio connects to."""

from __future__ import annotations

from dataclasses import dataclass

from langsmith_studio_nb._environment import detect_environment, needs_tunnel
from langsmith_studio_nb._runtime import Runtime, Worker
from langsmith_studio_nb._teardown import shut_down
from langsmith_studio_nb._urls import studio_url

DEFAULT_GRAPH_NAME = "agent"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 2024
DEFAULT_TIMEOUT = 180.0

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
    variable: str = DEFAULT_GRAPH_NAME,
    *,
    port: int = DEFAULT_PORT,
    tunnel: bool | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    runtime: Runtime | None = None,
) -> StudioSession:
    """Serve the agent named `variable` and show a link that opens it in Studio.

    Restarts any server this kernel already started, so re-running the cell picks
    up an edited agent.

    Args:
        variable: Name of the compiled graph in the notebook namespace.
        port: Port to serve on. A busy port is replaced with a free one.
        tunnel: Force a public tunnel on or off. Defaults to whether the
            notebook host runs the kernel away from the browser.
        timeout: Seconds to wait for the server to answer before giving up.
        runtime: Injected side effects. Tests substitute this.

    Raises:
        NameError: No such variable in the notebook namespace.
        RuntimeError: The server stopped while starting up.
        TimeoutError: The server did not answer within `timeout`.
    """
    global _active  # noqa: PLW0603 - one server per kernel, tracked at module scope

    runtime = runtime or Runtime()
    if variable not in runtime.namespace():
        message = (
            f"No variable named {variable!r} in the notebook. "
            "Run the cell that defines your agent first."
        )
        raise NameError(message)

    if tunnel is None:
        environment = detect_environment(modules=runtime.modules(), environ=runtime.environ)
        tunnel = needs_tunnel(environment)

    stop_studio(runtime=runtime)
    _active = runtime.spawn(
        lambda: runtime.run_server(
            host=DEFAULT_HOST,
            port=port,
            graphs={DEFAULT_GRAPH_NAME: f"__main__:{variable}"},
            tunnel=tunnel,
            reload=False,
            open_browser=False,
            allow_blocking=True,
        )
    )

    api_url = _wait_for_server(_active, runtime=runtime, timeout=timeout)
    url = studio_url(api_url, workspace_id=runtime.workspace_id())
    runtime.render(url)
    return StudioSession(api_url=api_url, studio_url=url, tunnel=tunnel)


def _wait_for_server(worker: Worker, *, runtime: Runtime, timeout: float) -> str:
    """Return the server's base URL once it answers, or raise."""
    deadline = runtime.now() + timeout
    while runtime.now() < deadline:
        if not worker.is_alive():
            message = "The agent server stopped while starting up. Check the log above."
            raise RuntimeError(message)
        api_url = runtime.environ.get(_API_URL_VARIABLE)
        if api_url and runtime.probe(f"{api_url}/ok"):
            return api_url
        runtime.sleep(_POLL_INTERVAL)
    message = f"The agent server did not answer within {timeout:g}s. Re-run this cell."
    raise TimeoutError(message)
