"""Find and shut down a previously started server and tunnel.

`run_server` owns `uvicorn.run` and never returns a handle, so a restart has to
recover one from the objects already alive in the kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


def is_uvicorn_server(candidate: Any) -> bool:  # noqa: ANN401 - scans arbitrary objects
    """Report whether `candidate` is a uvicorn server that can be asked to exit."""
    kind = type(candidate)
    return (
        kind.__name__ == "Server"
        and kind.__module__.split(".")[0] == "uvicorn"
        and hasattr(candidate, "should_exit")
    )


def is_cloudflared_process(candidate: Any) -> bool:  # noqa: ANN401 - scans arbitrary objects
    """Report whether `candidate` is a running cloudflared subprocess.

    Matched on type before any attribute is read: `getattr` against every object
    in the kernel triggers `__getattr__` side effects in unrelated libraries.
    """
    kind = type(candidate)
    if kind.__name__ != "Popen" or kind.__module__ != "subprocess":
        return False
    args = candidate.args
    text = " ".join(str(arg) for arg in args) if isinstance(args, (list, tuple)) else str(args)
    return "cloudflared" in text


def has_live_tunnel(objects: Iterable[Any]) -> bool:
    """Report whether a tunnel found in `objects` is still running."""
    return any(
        is_cloudflared_process(candidate) and candidate.poll() is None for candidate in objects
    )


def shut_down(objects: Iterable[Any], *, keep_tunnels: bool = False) -> int:
    """Stop every server and tunnel found in `objects`, returning how many were stopped.

    Keeping the tunnels leaves the public URL of a restarting server intact:
    a quick tunnel is rate limited per IP, so asking for a fresh one on every
    restart is what fails first on a shared notebook host.
    """
    stopped = 0
    for candidate in objects:
        if is_uvicorn_server(candidate):
            candidate.should_exit = True
            stopped += 1
        elif not keep_tunnels and is_cloudflared_process(candidate):
            candidate.kill()
            stopped += 1
    return stopped
