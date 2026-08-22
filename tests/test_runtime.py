import email.message
import logging
import socket
import sys
import threading
import types
import urllib.error
import urllib.request

import IPython.display
import pytest

from langsmith_studio_nb import _runtime
from langsmith_studio_nb._render import link_html, link_text
from langsmith_studio_nb._runtime import (
    Runtime,
    capture_uvicorn_server,
    default_display_html,
    default_find_free_port,
    default_in_colab,
    default_namespace,
    default_open_tunnel,
    default_port_is_free,
    default_quiet,
    default_render,
    default_run_server,
    default_spawn,
    default_status,
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
    worker.start()
    worker.join(timeout=5)

    assert done.is_set()
    assert worker.is_alive() is False


class FakeUvicornServer:
    """Stands in for uvicorn's own server object, built the same way."""

    def __init__(self, config=None):
        self.config = config
        self.should_exit = False


def _install_uvicorn(monkeypatch):
    return _install_module(monkeypatch, "uvicorn", Server=FakeUvicornServer)


def test_default_spawn_stops_the_server_its_thread_built(monkeypatch):
    uvicorn = _install_uvicorn(monkeypatch)
    built = []
    unowned = FakeUvicornServer()
    started = threading.Event()

    def run_until_stopped():
        server = uvicorn.Server(config="ours")  # what uvicorn.run does
        built.append(server)
        started.set()
        while not server.should_exit:
            threading.Event().wait(0.01)

    worker = default_spawn(run_until_stopped)
    worker.start()
    assert started.wait(5)
    worker.stop()
    worker.join(timeout=5)

    assert built[0].should_exit is True
    assert unowned.should_exit is False
    assert worker.is_alive() is False


def test_thread_worker_stops_a_server_that_is_still_starting(monkeypatch):
    """A stop must not slip past a server that does not exist yet."""
    uvicorn = _install_uvicorn(monkeypatch)
    stop_requested = threading.Event()
    built = []

    def build_late():
        assert stop_requested.wait(5)
        built.append(uvicorn.Server())

    worker = default_spawn(build_late)
    worker.start()
    worker.stop()
    stop_requested.set()
    worker.join(timeout=5)

    assert built[0].should_exit is True


def test_thread_worker_join_keeps_waiting_for_a_running_thread(monkeypatch):
    _install_uvicorn(monkeypatch)
    release = threading.Event()

    def wait_for_release() -> None:
        release.wait(5)

    worker = default_spawn(wait_for_release)
    worker.start()
    worker.join(timeout=0.01)

    assert worker.is_alive() is True

    release.set()
    worker.join(timeout=5)

    assert worker.is_alive() is False


def test_thread_worker_stop_is_safe_when_no_server_is_ever_built(monkeypatch):
    _install_uvicorn(monkeypatch)

    worker = default_spawn(lambda: None)
    worker.start()
    worker.join(timeout=5)
    worker.stop()

    assert worker.is_alive() is False


def test_capture_uvicorn_server_restores_uvicorn_after_one_server(monkeypatch):
    uvicorn = _install_uvicorn(monkeypatch)
    original = uvicorn.Server.__init__
    captured = []

    thread = capture_uvicorn_server(captured.append, target=lambda: uvicorn.Server(config="ours"))
    thread.start()
    thread.join(5)

    assert [server.config for server in captured] == ["ours"]
    assert uvicorn.Server.__init__ is original

    uvicorn.Server()

    assert len(captured) == 1


def test_capture_uvicorn_server_ignores_another_thread(monkeypatch):
    """The hook is process-wide; a server someone else builds is not ours to stop."""
    uvicorn = _install_uvicorn(monkeypatch)
    captured = []
    theirs = []

    capture_uvicorn_server(captured.append, target=lambda: None)
    elsewhere = threading.Thread(target=lambda: theirs.append(uvicorn.Server()))
    elsewhere.start()
    elsewhere.join(5)

    assert captured == []
    assert theirs[0].should_exit is False  # built normally, just not taken


def test_capture_uvicorn_server_unwinds_whatever_order_the_threads_finish(monkeypatch):
    """One dispatcher, not a stack: a worker that outlives its join pins nothing."""
    uvicorn = _install_uvicorn(monkeypatch)
    original = uvicorn.Server.__init__
    taken = {}
    go = threading.Event()

    def register(name, *, waits):
        def target():
            if waits:
                go.wait(5)
            uvicorn.Server(config=name)

        def sink(server):
            taken[name] = server.config

        return capture_uvicorn_server(sink, target=target)

    slow = register("slow", waits=True)  # registered first, builds last
    quick = register("quick", waits=False)
    quick.start()
    quick.join(5)

    assert taken == {"quick": "quick"}

    go.set()
    slow.start()
    slow.join(5)

    assert taken == {"quick": "quick", "slow": "slow"}
    assert uvicorn.Server.__init__ is original


def test_thread_worker_keeps_its_registration_when_the_thread_will_not_start(monkeypatch):
    """Deliberate: a kept registration fires for nobody, and a wrong guess loses a server."""
    uvicorn = _install_uvicorn(monkeypatch)
    original = uvicorn.Server.__init__

    def refuse(self):
        raise RuntimeError("can't start new thread")

    worker = default_spawn(lambda: None)
    monkeypatch.setattr(threading.Thread, "start", refuse)

    with pytest.raises(RuntimeError, match="can't start new thread"):
        worker.start()
    worker.join(timeout=5)  # what teardown does, and it must not guess either

    assert uvicorn.Server.__init__ is not original
    assert uvicorn.Server(config="theirs").config == "theirs"  # still built, just not taken


def test_thread_worker_registration_outlives_an_interrupted_start(monkeypatch):
    """The thread is running now and gives its own registration up; undoing here loses it."""
    _install_uvicorn(monkeypatch)
    release = threading.Event()
    real_start = threading.Thread.start

    def start_then_interrupt(self):
        real_start(self)
        raise KeyboardInterrupt

    def hold() -> None:
        release.wait(5)

    worker = default_spawn(hold)
    monkeypatch.setattr(threading.Thread, "start", start_then_interrupt)

    with pytest.raises(KeyboardInterrupt):
        worker.start()

    assert _runtime._capture_sinks != {}  # still waiting on the server this thread will build

    release.set()
    worker.join(5)

    assert _runtime._capture_sinks == {}


def test_default_open_tunnel_holds_no_exit_hook_when_cloudflared_will_not_start(monkeypatch):
    """The hook goes on before the child exists, so a spawn that fails has to take it off."""
    _, _, registered = _install_cloudflared(monkeypatch, BANNER)

    def refuse(command, **kwargs):
        raise OSError("no such binary")

    monkeypatch.setattr(_runtime.subprocess, "Popen", refuse)

    with pytest.raises(OSError, match="no such binary"):
        default_open_tunnel(2024)

    assert registered == []


def test_capture_uvicorn_server_drops_its_registration_when_the_thread_ends(monkeypatch):
    """A worker abandoned after its join would keep the dispatcher on Uvicorn otherwise."""
    uvicorn = _install_uvicorn(monkeypatch)
    original = uvicorn.Server.__init__

    abandoned = capture_uvicorn_server(lambda server: None, target=lambda: None)
    abandoned.start()
    abandoned.join(5)  # ran, died, never built a server, and nobody undid it

    assert uvicorn.Server.__init__ is original  # given up by the thread itself


class FakeCloudflared:
    """Stand-in for the cloudflared process, replaying what it prints."""

    def __init__(self, lines, status=0):
        self.stdout = iter(lines)
        self.status = status
        self.killed = False
        self.reaped = False

    def kill(self):
        self.killed = True

    def wait(self):
        self.reaped = True
        return self.status


BANNER = [
    "INF Requesting new quick Tunnel on trycloudflare.com...\n",
    "\n",
    "INF |  https://x.trycloudflare.com   |\n",
]


def _install_cloudflared(monkeypatch, lines, status=0):
    started = FakeCloudflared(lines, status)
    commands = []
    registered = []
    _install_module(
        monkeypatch, "langgraph_api.tunneling.cloudflare", ensure_cloudflared=lambda: "/bin/cfd"
    )

    def popen(command, **kwargs):
        commands.append(command)
        return started

    def register(func):
        registered.append(func)
        return func

    monkeypatch.setattr(_runtime.subprocess, "Popen", popen)
    monkeypatch.setattr(_runtime.atexit, "register", register)

    def unregister(func):
        if func in registered:  # the real one is a no-op for anything not registered
            registered.remove(func)

    monkeypatch.setattr(_runtime.atexit, "unregister", unregister)
    return started, commands, registered


def test_default_open_tunnel_returns_the_url_and_the_way_to_close_it(monkeypatch):
    started, commands, _ = _install_cloudflared(monkeypatch, BANNER)

    tunnel = default_open_tunnel(2024)

    assert tunnel.url == "https://x.trycloudflare.com"
    assert tunnel.port == 2024
    assert started.killed is False
    assert commands[0][:4] == ["/bin/cfd", "tunnel", "--url", "http://127.0.0.1:2024"]
    metrics = commands[0][commands[0].index("--metrics") + 1]
    assert tunnel.ready_url == f"http://{metrics}/ready"
    assert "--protocol" not in commands[0]

    tunnel.close()

    assert (started.killed, started.reaped) == (True, True)


def test_default_open_tunnel_exit_hook_kills_the_tunnel_it_went_on_for(monkeypatch):
    """The hook is in place before the child exists, so it has to find it afterwards."""
    started, _, registered = _install_cloudflared(monkeypatch, BANNER)
    default_open_tunnel(2024)

    registered[0]()  # what atexit runs on the way out

    assert started.killed is True


def test_default_open_tunnel_stops_holding_the_tunnels_it_closed(monkeypatch):
    """A demo restarts many times, and every exit hook left behind stays for the kernel."""
    started, _, registered = _install_cloudflared(monkeypatch, BANNER)

    for _ in range(3):
        started.stdout = iter(BANNER)
        default_open_tunnel(2024).close()

    assert registered == []


def test_default_open_tunnel_keeps_the_exit_hook_when_the_close_is_interrupted(monkeypatch):
    """Letting go before the process is reaped drops the last thing holding it."""
    started, _, registered = _install_cloudflared(monkeypatch, BANNER)
    tunnel = default_open_tunnel(2024)

    def interrupted():
        raise KeyboardInterrupt

    started.wait = interrupted

    with pytest.raises(KeyboardInterrupt):
        tunnel.close()

    assert registered != []  # cloudflared still dies with the kernel


def test_default_open_tunnel_close_is_safe_to_call_twice(monkeypatch):
    """Ownership hands a tunnel on before letting go, which can close it twice."""
    started, _, registered = _install_cloudflared(monkeypatch, BANNER)
    tunnel = default_open_tunnel(2024)

    tunnel.close()
    tunnel.close()

    assert (started.killed, started.reaped) == (True, True)
    assert registered == []


def test_default_open_tunnel_asks_for_a_protocol_when_given_one(monkeypatch):
    _, commands, _ = _install_cloudflared(monkeypatch, BANNER)

    default_open_tunnel(2024, protocol="http2")

    assert commands[0][-2:] == ["--protocol", "http2"]


def test_default_open_tunnel_reports_a_cloudflared_that_exits(monkeypatch):
    """Waiting out the timeout would report this as Cloudflare rate limiting."""
    started, _, registered = _install_cloudflared(
        monkeypatch, ["ERR failed to request quick Tunnel\n"], status=1
    )

    with pytest.raises(RuntimeError, match=r"exited with status 1\. ERR failed to request"):
        default_open_tunnel(2024, timeout=30.0)

    assert started.killed is True
    assert registered == []


class FailingStdout:
    """A pipe that breaks rather than ending, the way a dead read does."""

    def __init__(self, lines=()):
        self._lines = lines

    def __iter__(self):
        yield from self._lines
        raise OSError("pipe went away")


def test_default_open_tunnel_reports_a_reader_that_fails(monkeypatch):
    """An unsettled future would strand the caller on its timeout and blame Cloudflare."""
    started, _, _ = _install_cloudflared(monkeypatch, [])
    started.stdout = FailingStdout()

    with pytest.raises(OSError, match="pipe went away"):
        default_open_tunnel(2024, timeout=30.0)

    assert started.killed is True


def test_default_open_tunnel_keeps_a_url_a_later_read_error_cannot_take_back(monkeypatch):
    """The tunnel is up; losing the log stream afterwards is not a reason to drop it."""
    started, _, _ = _install_cloudflared(monkeypatch, [])
    started.stdout = FailingStdout(BANNER)

    tunnel = default_open_tunnel(2024)

    assert tunnel.url == "https://x.trycloudflare.com"
    assert started.killed is False


class BlockingStdout:
    """A pipe that says its piece and then stays open and silent, like a stuck cloudflared."""

    def __init__(self, released, lines=()):
        self._released = released
        self._lines = lines

    def __iter__(self):
        yield from self._lines
        self._released.wait(5)


def test_default_open_tunnel_kills_its_process_when_no_url_arrives(monkeypatch):
    """A process that says nothing and stays up has to be timed out."""
    released = threading.Event()
    started, _, registered = _install_cloudflared(monkeypatch, [])
    started.stdout = BlockingStdout(released)

    try:
        with pytest.raises(TimeoutError):
            default_open_tunnel(2024, timeout=0.2)
    finally:
        released.set()

    assert started.killed is True
    assert started.reaped is True
    assert registered == []


def test_default_open_tunnel_timeout_quotes_cloudflared(monkeypatch):
    """Blaming the rate limit for every silent tunnel buries what cloudflared reported."""
    released = threading.Event()
    started, _, _ = _install_cloudflared(monkeypatch, [])
    started.stdout = BlockingStdout(released, ["INF Requesting new quick Tunnel\n"])

    try:
        with pytest.raises(TimeoutError, match="Requesting new quick Tunnel"):
            default_open_tunnel(2024, timeout=0.2)
    finally:
        released.set()


def test_default_open_tunnel_logs_what_cloudflared_says(monkeypatch, caplog):
    _install_cloudflared(monkeypatch, BANNER)

    with caplog.at_level(logging.INFO, logger=_runtime.TUNNEL_LOGGER):
        default_open_tunnel(2024)

    assert any("Requesting new quick Tunnel" in record.message for record in caplog.records)


@pytest.mark.parametrize("status", [200, 503])
def test_default_status(monkeypatch, status):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: FakeResponse(status))

    assert default_status("http://127.0.0.1:2024/ok") == status


def test_default_status_reports_an_http_error_as_a_status(monkeypatch):
    """Cloudflare answering 530 for a dead tunnel is a verdict, not a failure to reach."""

    def error(*args, **kwargs):
        raise urllib.error.HTTPError(
            "https://x.trycloudflare.com/ok", 530, "", email.message.Message(), None
        )

    monkeypatch.setattr(urllib.request, "urlopen", error)

    assert default_status("https://x.trycloudflare.com/ok") == 530


def test_default_status_when_nothing_answers(monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)

    assert default_status("http://127.0.0.1:2024/ok") is None


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

    assert shown == [link_html("https://studio.example", hint="add the domain")]
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


def test_default_in_colab(monkeypatch):
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)

    assert default_in_colab() is False

    _install_module(monkeypatch, "google.colab", userdata=None)

    assert default_in_colab() is True


def test_runtime_defaults_are_the_real_implementations():
    runtime = Runtime()

    assert runtime.run_server is default_run_server
    assert runtime.open_tunnel is default_open_tunnel
    assert runtime.status is default_status
    assert runtime.port_is_free is default_port_is_free
    assert runtime.in_colab is default_in_colab
    assert runtime.find_free_port is default_find_free_port
    assert runtime.environ is not None
