"""Fakes for every side effect `start_studio` performs."""

from __future__ import annotations

from typing import Any

import pytest

from langsmith_studio_nb import _session
from langsmith_studio_nb._runtime import Runtime


class FakeWorker:
    """Stand-in for the thread running the agent server."""

    def __init__(self, *, alive: bool = True) -> None:
        self.alive = alive
        self.joins: list[float | None] = []

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        self.joins.append(timeout)


class FakeRuntime:
    """Builds a `Runtime` whose effects are recorded instead of performed."""

    def __init__(
        self,
        *,
        namespace: dict[str, Any] | None = None,
        environ: dict[str, str] | None = None,
        modules: tuple[str, ...] = (),
        probes: list[bool] | None = None,
        worker: FakeWorker | None = None,
        workspace_id: str | None = "ws-1",
        live_objects: list[Any] | None = None,
        api_url: str | None = "http://127.0.0.1:2024",
        tick: float = 0.1,
    ) -> None:
        self.namespace_value = {"agent": object()} if namespace is None else namespace
        self.environ = dict(environ or {})
        self.modules_value = modules
        self.probes = probes
        self.worker = worker or FakeWorker()
        self.workspace_id_value = workspace_id
        self.live_objects_value = live_objects or []
        self.api_url = api_url
        self.tick = tick

        self.clock = 0.0
        self.server_calls: list[dict[str, Any]] = []
        self.rendered: list[str] = []
        self.sleeps: list[float] = []

    def run_server(self, **kwargs: Any) -> None:
        self.server_calls.append(kwargs)

    def spawn(self, target: Any) -> FakeWorker:
        target()  # the real spawn runs this on a thread; run it here to record the call
        if self.api_url is not None:
            self.environ["LANGGRAPH_API_URL"] = self.api_url
        return self.worker

    def probe(self, url: str) -> bool:
        if self.probes is None:
            return True
        return self.probes.pop(0) if self.probes else False

    def now(self) -> float:
        self.clock += self.tick
        return self.clock

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    def render(self, url: str) -> None:
        self.rendered.append(url)

    def build(self) -> Runtime:
        return Runtime(
            run_server=self.run_server,
            spawn=self.spawn,
            probe=self.probe,
            workspace_id=lambda: self.workspace_id_value,
            namespace=lambda: self.namespace_value,
            modules=lambda: self.modules_value,
            live_objects=lambda: self.live_objects_value,
            render=self.render,
            sleep=self.sleep,
            now=self.now,
            environ=self.environ,
        )


@pytest.fixture(autouse=True)
def _no_active_server() -> None:
    """Keep the module-level server handle from leaking between tests."""
    _session._active = None
