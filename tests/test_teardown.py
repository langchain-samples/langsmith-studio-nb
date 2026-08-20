from langsmith_studio_nb._teardown import (
    has_live_tunnel,
    is_cloudflared_process,
    is_uvicorn_server,
    shut_down,
)


class Server:
    def __init__(self) -> None:
        self.should_exit = False


Server.__module__ = "uvicorn.server"


class Impostor:
    """Named Server, but not uvicorn's."""

    should_exit = False


Impostor.__name__ = "Server"


class Process:
    def __init__(self, args, returncode=None) -> None:
        self.args = args
        self.returncode = returncode
        self.killed = False

    def poll(self):
        return self.returncode

    def kill(self) -> None:
        self.killed = True


Process.__name__ = "Popen"
Process.__module__ = "subprocess"


def test_is_uvicorn_server():
    assert is_uvicorn_server(Server()) is True
    assert is_uvicorn_server(Impostor()) is False
    assert is_uvicorn_server(object()) is False


def test_is_cloudflared_process():
    assert is_cloudflared_process(Process(["/bin/cloudflared", "tunnel"])) is True
    assert is_cloudflared_process(Process("cloudflared tunnel --url x")) is True
    assert is_cloudflared_process(Process(["/bin/other"])) is False
    assert is_cloudflared_process(object()) is False


def test_is_cloudflared_process_ignores_look_alikes():
    class NotAProcess:
        args = ["cloudflared"]

    assert is_cloudflared_process(NotAProcess()) is False


def test_is_cloudflared_process_does_not_probe_attributes():
    """A gc sweep must not wake up objects that synthesize attributes on access."""

    class Trap:
        def __getattr__(self, name):
            raise AssertionError(f"attribute {name!r} was probed")

    assert is_cloudflared_process(Trap()) is False


def test_shut_down_stops_servers_and_tunnels():
    server = Server()
    tunnel = Process(["cloudflared", "tunnel"])
    bystander = Process(["sleep", "1"])

    stopped = shut_down([server, tunnel, bystander, object()])

    assert stopped == 2
    assert server.should_exit is True
    assert tunnel.killed is True
    assert bystander.killed is False


def test_shut_down_with_nothing_running():
    assert shut_down([]) == 0


def test_shut_down_can_keep_the_tunnels():
    server = Server()
    tunnel = Process(["cloudflared", "tunnel"])

    stopped = shut_down([server, tunnel], keep_tunnels=True)

    assert stopped == 1
    assert server.should_exit is True
    assert tunnel.killed is False


def test_has_live_tunnel():
    assert has_live_tunnel([Process(["cloudflared", "tunnel"])]) is True
    assert has_live_tunnel([Process(["cloudflared", "tunnel"], returncode=0)]) is False
    assert has_live_tunnel([Process(["sleep", "1"]), object()]) is False
    assert has_live_tunnel([]) is False
