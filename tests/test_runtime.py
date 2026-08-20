import logging
import socket
import sys
import threading
import types
import urllib.request

import IPython.display
import pytest

from langsmith_studio_nb import _runtime
from langsmith_studio_nb._render import link_text
from langsmith_studio_nb._runtime import (
    Runtime,
    _request_server_exit,
    _ThreadWorker,
    default_display_html,
    default_find_free_port,
    default_modules,
    default_namespace,
    default_open_tunnel,
    default_port_is_free,
    default_probe,
    default_quiet,
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


def test_default_spawn_stops_only_the_server_on_its_worker_thread():
    class Server:
        def __init__(self):
            self.should_exit = False

    Server.__module__ = "uvicorn.server"
    owned = Server()
    unowned = Server()
    started = threading.Event()

    def run_until_stopped():
        server = owned
        started.set()
        while not server.should_exit:
            threading.Event().wait(0.01)

    worker = default_spawn(run_until_stopped)
    assert started.wait(5)
    worker.stop()
    worker.join(timeout=5)

    assert owned.should_exit is True
    assert unowned.should_exit is False
    assert worker.is_alive() is False


def test_request_server_exit_is_safe_before_a_server_exists():
    _request_server_exit(None)


def test_thread_worker_stop_is_safe_before_the_thread_has_an_identity(monkeypatch):
    worker = object.__new__(_ThreadWorker)
    monkeypatch.setattr(worker, "_thread", types.SimpleNamespace(ident=None), raising=False)

    worker.stop()


def test_default_open_tunnel_returns_the_url_and_exact_process(monkeypatch):
    class Future:
        def result(self, *, timeout):
            assert timeout == 30.0
            return "https://x.trycloudflare.com"

    class Process:
        killed = False

        def kill(self):
            self.killed = True

    process = Process()
    cloudflare = _install_module(
        monkeypatch,
        "langgraph_api.tunneling.cloudflare",
        start_tunnel=lambda port: types.SimpleNamespace(url=Future(), process=process),
    )
    assert cloudflare is not None

    tunnel = default_open_tunnel(2024)

    assert tunnel.url == "https://x.trycloudflare.com"
    assert tunnel.process is process
    assert process.killed is False


def test_default_open_tunnel_kills_its_process_when_startup_fails(monkeypatch):
    class Future:
        def result(self, *, timeout):
            raise TimeoutError

    class Process:
        killed = False

        def kill(self):
            self.killed = True

    process = Process()
    _install_module(
        monkeypatch,
        "langgraph_api.tunneling.cloudflare",
        start_tunnel=lambda port: types.SimpleNamespace(url=Future(), process=process),
    )

    with pytest.raises(TimeoutError):
        default_open_tunnel(2024)

    assert process.killed is True


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


def test_default_port_is_free():
    port = default_find_free_port()

    assert default_port_is_free(port) is True


def test_default_port_is_free_reports_a_port_in_use():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)

        assert default_port_is_free(taken.getsockname()[1]) is False


def test_default_namespace_is_the_notebook_namespace():
    assert default_namespace() is sys.modules["__main__"].__dict__


def test_default_modules_lists_imported_modules():
    assert "sys" in default_modules()


def test_default_display_html(monkeypatch):
    shown = []
    monkeypatch.setattr(IPython.display, "display", shown.append)

    assert default_display_html("<b>hi</b>") is True
    assert shown[0].data == "<b>hi</b>"


def test_default_display_html_without_ipython(monkeypatch):
    """The package must not require IPython; Colab pins an old one."""
    monkeypatch.setitem(sys.modules, "IPython.display", None)

    assert default_display_html("<b>hi</b>") is False


def test_default_render_displays_the_button_only(monkeypatch, capsys):
    shown = []

    def display(html):
        shown.append(html)
        return True

    monkeypatch.setattr(_runtime, "default_display_html", display)

    default_render("https://studio.example", "add the domain")

    assert "https://studio.example" in shown[0]
    assert "add the domain" in shown[0]
    assert capsys.readouterr().out == ""


def test_default_render_falls_back_to_text(monkeypatch, capsys):
    monkeypatch.setattr(_runtime, "default_display_html", lambda html: False)

    default_render("https://studio.example", "add the domain")

    assert capsys.readouterr().out.strip() == link_text(
        "https://studio.example", hint="add the domain"
    )


def test_default_quiet_silences_the_server(monkeypatch):
    levels = {}

    def restore():
        return None

    monkeypatch.setattr(
        _runtime,
        "silence_loggers",
        lambda level: (levels.setdefault("level", level), restore)[1],
    )

    returned = default_quiet()

    assert levels["level"] == logging.ERROR
    assert returned is restore


def test_runtime_defaults_are_the_real_implementations():
    runtime = Runtime()

    assert runtime.run_server is default_run_server
    assert runtime.open_tunnel is default_open_tunnel
    assert runtime.probe is default_probe
    assert runtime.port_is_free is default_port_is_free
    assert runtime.find_free_port is default_find_free_port
    assert runtime.environ is not None
