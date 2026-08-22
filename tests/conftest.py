"""Fakes for every side effect `start_studio` performs."""

from __future__ import annotations

from typing import Any

import pytest

from langsmith_studio_nb import _runtime, _session
from langsmith_studio_nb._runtime import OpenedTunnel, Runtime


class FakeWorker:
    """Stand-in for the thread running the agent server."""

    def __init__(
        self,
        *,
        alive: bool = True,
        stops: bool = True,
        stop_error: BaseException | None = None,
        join_error: BaseException | None = None,
        start_error: BaseException | None = None,
    ) -> None:
        self.alive = alive
        self.stops = stops
        self.stop_error = stop_error
        self.join_error = join_error
        self.start_error = start_error
        self.target: Any = None
        self.joins: list[float | None] = []
        self.starts = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.starts += 1
        if self.start_error is not None:
            raise self.start_error
        if self.stop_calls and self.stops:
            self.alive = True  # a restart brings this one back
        if self.target is not None:
            self.target()  # the real worker runs this on a thread

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        self.joins.append(timeout)
        if self.join_error is not None:
            raise self.join_error

    def stop(self) -> None:
        self.stop_calls += 1
        if self.stop_error is not None:
            raise self.stop_error
        if self.stops:
            self.alive = False


class FakeTunnel:
    """Stand-in for the cloudflared process one session owns."""

    def __init__(self) -> None:
        self.closes = 0
        self.close_error: BaseException | None = None

    @property
    def killed(self) -> bool:
        return self.closes > 0

    def close(self) -> None:
        self.closes += 1
        if self.close_error is not None:
            raise self.close_error


class FakeRuntime:
    """Build a `Runtime` that records its effects instead of performing them."""

    def __init__(
        self,
        *,
        namespace: dict[str, Any] | None = None,
        environ: dict[str, str] | None = None,
        in_colab: bool = False,
        statuses: list[int | None] | None = None,
        status_error: BaseException | None = None,
        unroutable: tuple[str, ...] = (),
        unreachable: tuple[str, ...] = (),
        worker: FakeWorker | None = None,
        spawn_error: Exception | None = None,
        workspace_id: str | None = "ws-1",
        tunnel_url: str = "https://x.trycloudflare.com",
        tunnel_urls: list[str] | None = None,
        tunnel_ready: list[bool] | None = None,
        tunnel_error: Exception | None = None,
        api_url: str | None = None,
        publishes_url: bool = True,
        port_is_free: bool = True,
        free_port: int = 51234,
        tick: float = 0.1,
    ) -> None:
        self.namespace_value = {"agent": object()} if namespace is None else namespace
        self.environ = dict(environ or {})
        self.in_colab_value = in_colab
        self.statuses = statuses
        self.status_error = status_error
        self.unroutable = unroutable
        self.unreachable = unreachable
        self.worker = worker or FakeWorker()
        self.spawn_error = spawn_error
        self.workspace_id_value = workspace_id
        self.tunnel_url = tunnel_url
        self.tunnel_urls = tunnel_urls
        self.tunnel_ready = tunnel_ready
        self.tunnel_error = tunnel_error
        self.api_url = api_url
        self.publishes_url = publishes_url
        self.port_is_free_value = port_is_free
        self.free_port = free_port
        self.tick = tick

        self.clock = 0.0
        self.server_calls: list[dict[str, Any]] = []
        self.tunnels: list[FakeTunnel] = []
        self.ready_urls: list[str] = []
        self.connected: dict[str, bool] = {}
        self.opened_tunnel_ports: list[int] = []
        self.opened_tunnel_protocols: list[str | None] = []
        self.rendered: list[tuple[str, str | None]] = []
        self.sleeps: list[float] = []
        self.probed: list[str] = []
        self.after_probe: Any = None
        self.quieted = 0
        self.logging_restored = 0
        self.restore_error: BaseException | None = None

    def run_server(self, **kwargs: Any) -> None:
        self.server_calls.append(kwargs)
        if self.publishes_url:  # the real server publishes the URL it bound
            self.environ["LANGGRAPH_API_URL"] = self.api_url or f"http://127.0.0.1:{kwargs['port']}"

    def spawn(self, target: Any) -> FakeWorker:
        """Build a worker without starting it, the way the real one does."""
        if self.spawn_error is not None:
            raise self.spawn_error
        self.worker.target = target
        return self.worker

    def open_tunnel(self, port: int, *, protocol: str | None = None) -> OpenedTunnel:
        self.opened_tunnel_ports.append(port)
        self.opened_tunnel_protocols.append(protocol)
        if self.tunnel_error is not None:
            raise self.tunnel_error
        index = len(self.tunnels)
        tunnel = FakeTunnel()
        url = self.tunnel_url
        if self.tunnel_urls:
            url = self.tunnel_urls[min(index, len(self.tunnel_urls) - 1)]
        ready_url = f"http://127.0.0.1:{20241 + index}/ready"
        self.tunnels.append(tunnel)
        self.ready_urls.append(ready_url)
        self.connected[ready_url] = (
            self.tunnel_ready[index]
            if self.tunnel_ready and index < len(self.tunnel_ready)
            else True
        )
        return OpenedTunnel(url=url, port=port, ready_url=ready_url, close=tunnel.close)

    def drop_tunnel(self, index: int) -> None:
        """Model Cloudflare dropping a tunnel the process is still retrying."""
        self.connected[self.ready_urls[index]] = False

    def status(self, url: str) -> int | None:
        """Report a status the way the real one does. None means nothing answered."""
        self.probed.append(url)
        answer = self._status(url)
        if self.after_probe is not None:
            self.after_probe(url)  # model something changing between two probes
        return answer

    def _status(self, url: str) -> int | None:
        if url in self.connected:  # cloudflared's own health check
            if self.status_error is not None:
                raise self.status_error
            return 200 if self.connected[url] else 503
        if any(url.startswith(prefix) for prefix in self.unroutable):
            return 530  # Cloudflare answering for a tunnel it cannot route
        if any(url.startswith(prefix) for prefix in self.unreachable):
            return None
        if self.statuses is None:
            return 200
        return self.statuses.pop(0) if self.statuses else None

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
            if self.restore_error is not None:
                raise self.restore_error

        return restore

    def build(self) -> Runtime:
        return Runtime(
            run_server=self.run_server,
            spawn=self.spawn,
            open_tunnel=self.open_tunnel,
            status=self.status,
            port_is_free=lambda _port: self.port_is_free_value,
            find_free_port=lambda: self.free_port,
            workspace_id=lambda: self.workspace_id_value,
            namespace=lambda: self.namespace_value,
            in_colab=lambda: self.in_colab_value,
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


@pytest.fixture(autouse=True)
def _no_uvicorn_capture() -> None:
    """Drop registrations for a fake `uvicorn` the next test will not have."""
    _runtime._capture_sinks.clear()
    _runtime._capture_installed = None
