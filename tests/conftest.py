"""Fakes for every side effect `start_studio` performs."""

from __future__ import annotations

from typing import Any

import pytest

from langsmith_studio_nb import _session
from langsmith_studio_nb._runtime import OpenedTunnel, Runtime


class FakeWorker:
    """Stand-in for the thread running the agent server."""

    def __init__(self, *, alive: bool = True, stops: bool = True) -> None:
        self.alive = alive
        self.stops = stops
        self.joins: list[float | None] = []
        self.stop_calls = 0

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        self.joins.append(timeout)

    def stop(self) -> None:
        self.stop_calls += 1
        if self.stops:
            self.alive = False


class FakeTunnelProcess:
    """Stand-in for the exact cloudflared process owned by a session."""

    def __init__(self) -> None:
        self.killed = False

    def kill(self) -> None:
        self.killed = True


class FakeRuntime:
    """Builds a `Runtime` whose effects are recorded instead of performed."""

    def __init__(
        self,
        *,
        namespace: dict[str, Any] | None = None,
        environ: dict[str, str] | None = None,
        modules: tuple[str, ...] = (),
        probes: list[bool] | None = None,
        unreachable: tuple[str, ...] = (),
        worker: FakeWorker | None = None,
        spawn_error: Exception | None = None,
        workspace_id: str | None = "ws-1",
        tunnel_url: str = "https://x.trycloudflare.com",
        tunnel_error: Exception | None = None,
        port_is_free: bool = True,
        free_port: int = 51234,
        tick: float = 0.1,
    ) -> None:
        self.namespace_value = {"agent": object()} if namespace is None else namespace
        self.environ = dict(environ or {})
        self.modules_value = modules
        self.probes = probes
        self.unreachable = unreachable
        self.worker = worker or FakeWorker()
        self.spawn_error = spawn_error
        self.workspace_id_value = workspace_id
        self.tunnel_url = tunnel_url
        self.tunnel_error = tunnel_error
        self.port_is_free_value = port_is_free
        self.free_port = free_port
        self.tick = tick

        self.clock = 0.0
        self.server_calls: list[dict[str, Any]] = []
        self.tunnels: list[FakeTunnelProcess] = []
        self.opened_tunnel_ports: list[int] = []
        self.rendered: list[tuple[str, str | None]] = []
        self.sleeps: list[float] = []
        self.probed: list[str] = []
        self.quieted = 0
        self.logging_restored = 0

    def run_server(self, **kwargs: Any) -> None:
        self.server_calls.append(kwargs)

    def spawn(self, target: Any) -> FakeWorker:
        if self.spawn_error is not None:
            raise self.spawn_error
        target()  # the real spawn runs this on a thread; run it here to record the call
        if self.worker.stop_calls and self.worker.stops:
            self.worker.alive = True
        return self.worker

    def open_tunnel(self, port: int) -> OpenedTunnel:
        self.opened_tunnel_ports.append(port)
        if self.tunnel_error is not None:
            raise self.tunnel_error
        process = FakeTunnelProcess()
        self.tunnels.append(process)
        return OpenedTunnel(url=self.tunnel_url, process=process)

    def probe(self, url: str) -> bool:
        self.probed.append(url)
        if any(url.startswith(prefix) for prefix in self.unreachable):
            return False
        if self.probes is None:
            return True
        return self.probes.pop(0) if self.probes else False

    def now(self) -> float:
        self.clock += self.tick
        return self.clock

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    def render(self, url: str, hint: str | None = None) -> None:
        self.rendered.append((url, hint))

    def quiet(self):
        self.quieted += 1

        def restore() -> None:
            self.logging_restored += 1

        return restore

    def build(self) -> Runtime:
        return Runtime(
            run_server=self.run_server,
            spawn=self.spawn,
            open_tunnel=self.open_tunnel,
            probe=self.probe,
            port_is_free=lambda _port: self.port_is_free_value,
            find_free_port=lambda: self.free_port,
            workspace_id=lambda: self.workspace_id_value,
            namespace=lambda: self.namespace_value,
            modules=lambda: self.modules_value,
            render=self.render,
            quiet=self.quiet,
            sleep=self.sleep,
            now=self.now,
            environ=self.environ,
        )


@pytest.fixture(autouse=True)
def _no_active_server() -> None:
    """Keep the module-level server and tunnel handles from leaking between tests."""
    _session._state = None
