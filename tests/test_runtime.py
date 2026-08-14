import sys
import threading
import types
import urllib.request

import IPython.display
import pytest

from langsmith_studio_nb import _runtime
from langsmith_studio_nb._runtime import (
    Runtime,
    default_display_html,
    default_live_objects,
    default_modules,
    default_namespace,
    default_probe,
    default_render,
    default_run_server,
    default_spawn,
    default_workspace_id,
)


class FakeResponse:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _install_module(monkeypatch, name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_default_run_server_delegates_to_langgraph(monkeypatch):
    calls = []
    cli = _install_module(
        monkeypatch, "langgraph_api.cli", run_server=lambda **kw: calls.append(kw)
    )
    _install_module(monkeypatch, "langgraph_api", cli=cli)

    default_run_server(port=2024, tunnel=True)

    assert calls == [{"port": 2024, "tunnel": True}]


def test_default_spawn_runs_the_target_off_thread():
    done = threading.Event()

    worker = default_spawn(done.set)
    worker.join(timeout=5)

    assert done.is_set()
    assert worker.is_alive() is False


@pytest.mark.parametrize(("status", "expected"), [(200, True), (503, False)])
def test_default_probe(monkeypatch, status, expected):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: FakeResponse(status))

    assert default_probe("http://127.0.0.1:2024/ok") is expected


def test_default_probe_when_the_server_is_not_listening(monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)

    assert default_probe("http://127.0.0.1:2024/ok") is False


def _langsmith_client(sessions):
    class Response:
        def json(self):
            return sessions

    class Client:
        def request_with_retries(self, *args, **kwargs):
            return Response()

    return Client


def test_default_workspace_id(monkeypatch):
    _install_module(monkeypatch, "langsmith", Client=_langsmith_client([{"tenant_id": "ws-1"}]))

    assert default_workspace_id() == "ws-1"


def test_default_workspace_id_without_sessions(monkeypatch):
    _install_module(monkeypatch, "langsmith", Client=_langsmith_client([]))

    assert default_workspace_id() is None


def test_default_workspace_id_when_the_request_fails(monkeypatch):
    class Client:
        def request_with_retries(self, *args, **kwargs):
            raise ValueError("unauthorized")

    _install_module(monkeypatch, "langsmith", Client=Client)

    assert default_workspace_id() is None


def test_default_workspace_id_without_langsmith_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "langsmith", None)

    assert default_workspace_id() is None


def test_default_namespace_is_the_notebook_namespace():
    assert default_namespace() is sys.modules["__main__"].__dict__


def test_default_modules_lists_imported_modules():
    assert "sys" in default_modules()


def test_default_live_objects_includes_this_object():
    marker = ["langsmith_studio_nb-marker"]  # object() is not tracked by the garbage collector

    assert any(candidate is marker for candidate in default_live_objects())


def test_default_display_html(monkeypatch):
    shown = []
    monkeypatch.setattr(IPython.display, "display", shown.append)

    default_display_html("<b>hi</b>")

    assert shown[0].data == "<b>hi</b>"


def test_default_render_displays_and_prints(monkeypatch, capsys):
    shown = []
    monkeypatch.setattr(_runtime, "default_display_html", shown.append)

    default_render("https://studio.example")

    assert "https://studio.example" in shown[0]
    assert capsys.readouterr().out.strip() == "https://studio.example"


def test_runtime_defaults_are_the_real_implementations():
    runtime = Runtime()

    assert runtime.run_server is default_run_server
    assert runtime.probe is default_probe
    assert runtime.environ is not None
