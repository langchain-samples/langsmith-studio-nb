"""What `start_studio` decides before it touches anything.

Pure functions over plain values, so the decisions can be read and tested
without a server, a thread, or a tunnel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Container, Sequence

DEFAULT_GRAPH_NAME = "agent"
DEFAULT_PORT = 2024
DEFAULT_TIMEOUT = 180.0

# Every attempt spends a quick tunnel against a rate limit that every notebook
# on this host shares, so every attempt has to try a route the last one did not.
# The default reaches Cloudflare's port 7844 over UDP and `http2` reaches the
# same port over TCP; a runtime may drop one and not the other. Do not add a
# repeat of either: it buys no new route and spends the quota anyway.
TUNNEL_PROTOCOLS: tuple[str | None, ...] = (None, "http2")


@dataclass(frozen=True)
class Request:
    """What one `start_studio` call asks for, once its defaults are resolved."""

    graphs: tuple[str, ...]
    port: int
    tunnel: bool


def should_tunnel(explicit: bool | None, *, in_colab: bool) -> bool:
    """Decide whether Studio needs a public URL to reach the server.

    Studio runs in the browser, so it can only reach localhost when the kernel
    is on the same machine.
    """
    return in_colab if explicit is None else explicit


def missing_variables(graphs: Sequence[str], namespace: Container[str]) -> tuple[str, ...]:
    """Return the names that the notebook does not define, in the order asked for."""
    return tuple(name for name in graphs if name not in namespace)


def tunnel_still_reaches(*, tunnel_port: int, bound_port: int) -> bool:
    """Report whether a tunnel opened on `tunnel_port` still reaches the server.

    A tunnel forwards to one port and cannot follow a server that moved.
    """
    return tunnel_port == bound_port


def plan(
    variables: Sequence[str],
    *,
    port: int,
    tunnel: bool,
    namespace: Container[str],
) -> Request:
    """Resolve one call's arguments into the request it stands for.

    Raises:
        NameError: A named variable is not defined in the notebook.
    """
    graphs = tuple(variables) or (DEFAULT_GRAPH_NAME,)
    missing = missing_variables(graphs, namespace)
    if missing:
        message = (
            f"No variable named {missing[0]!r} in the notebook. "
            "Run the cell that defines your agent first."
        )
        raise NameError(message)
    return Request(graphs=graphs, port=port, tunnel=tunnel)
