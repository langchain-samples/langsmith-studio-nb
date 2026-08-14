import pytest

from langsmith_studio_nb import _session
from langsmith_studio_nb._session import (
    StudioSession,
    readiness_url,
    start_studio,
    stop_studio,
)
from tests.conftest import FakeRuntime, FakeWorker


def test_start_studio_serves_the_notebook_agent_and_returns_the_studio_url():
    fake = FakeRuntime()

    session = start_studio(runtime=fake.build())

    assert session == StudioSession(
        api_url="http://127.0.0.1:2024",
        studio_url=(
            "https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024&organizationId=ws-1"
        ),
        tunnel=False,
    )
    assert fake.rendered == [(session.studio_url, None)]
    assert fake.server_calls[0]["graphs"] == {"agent": "__main__:agent"}


def test_start_studio_tunnels_when_the_kernel_is_remote():
    fake = FakeRuntime(modules=("google.colab",))

    session = start_studio(runtime=fake.build())

    assert session.tunnel is True
    assert fake.server_calls[0]["tunnel"] is True


def test_start_studio_honors_an_explicit_tunnel_choice():
    fake = FakeRuntime(modules=("google.colab",))

    session = start_studio(runtime=fake.build(), tunnel=False)

    assert session.tunnel is False
    assert fake.server_calls[0]["tunnel"] is False


def test_start_studio_serves_a_named_variable_on_a_chosen_port():
    fake = FakeRuntime(namespace={"researcher": object()})

    start_studio("researcher", port=8123, runtime=fake.build())

    assert fake.server_calls[0]["graphs"] == {"agent": "__main__:researcher"}
    assert fake.server_calls[0]["port"] == 8123


def test_start_studio_without_a_workspace_omits_the_organization():
    fake = FakeRuntime(workspace_id=None)

    session = start_studio(runtime=fake.build())

    assert "organizationId" not in session.studio_url


def test_start_studio_waits_for_the_server_to_answer():
    fake = FakeRuntime(probes=[False, True])

    start_studio(runtime=fake.build())

    assert fake.sleeps == [0.5]


def test_start_studio_rejects_an_undefined_variable():
    fake = FakeRuntime(namespace={})

    with pytest.raises(NameError, match="Run the cell that defines your agent first"):
        start_studio(runtime=fake.build())

    assert fake.server_calls == []


def test_start_studio_reports_a_server_that_died_on_startup():
    fake = FakeRuntime(worker=FakeWorker(alive=False))

    with pytest.raises(RuntimeError, match="stopped while starting up"):
        start_studio(runtime=fake.build())


def test_start_studio_times_out_when_the_url_never_appears():
    fake = FakeRuntime(api_url=None, tick=0.3)

    with pytest.raises(TimeoutError, match="did not answer within 1s"):
        start_studio(runtime=fake.build(), timeout=1.0)

    assert fake.sleeps == [0.5, 0.5, 0.5]


def test_start_studio_restarts_a_running_server():
    fake = FakeRuntime()
    runtime = fake.build()

    start_studio(runtime=runtime)
    start_studio(runtime=runtime)

    assert fake.worker.joins == [20.0]
    assert len(fake.server_calls) == 2


def test_stop_studio_clears_the_server_url_and_handle():
    fake = FakeRuntime()
    runtime = fake.build()
    start_studio(runtime=runtime)

    stop_studio(runtime=runtime)

    assert "LANGGRAPH_API_URL" not in fake.environ
    assert _session._active is None


def test_stop_studio_shuts_down_what_it_finds():
    class Server:
        should_exit = False

    Server.__module__ = "uvicorn.server"
    server = Server()
    fake = FakeRuntime(live_objects=[server])

    stop_studio(runtime=fake.build())

    assert server.should_exit is True


def test_stop_studio_is_safe_with_no_server_running():
    stop_studio()

    assert _session._active is None


def test_start_studio_silences_the_server_by_default():
    fake = FakeRuntime()

    start_studio(runtime=fake.build())

    assert fake.quieted == 1
    assert fake.server_calls[0]["server_level"] == "ERROR"


def test_start_studio_verbose_keeps_the_server_logs():
    fake = FakeRuntime()

    start_studio(runtime=fake.build(), verbose=True)

    assert fake.quieted == 0
    assert fake.server_calls[0]["server_level"] == "INFO"


def test_start_studio_hints_at_allowed_domains_when_tunneling():
    fake = FakeRuntime(modules=("google.colab",))

    start_studio(runtime=fake.build())

    _, hint = fake.rendered[0]
    assert hint is not None
    assert "*.trycloudflare.com" in hint


def test_session_renders_as_nothing_when_echoed():
    """The link is already displayed; the dataclass repr would duplicate it."""
    session = StudioSession(api_url="http://x", studio_url="http://y", tunnel=False)

    assert session._repr_html_() == ""


def test_readiness_url_polls_the_local_server_when_tunneling():
    """A tunnel hostname may not resolve from the kernel; only the browser matters."""
    url = readiness_url("https://x.trycloudflare.com", port=2024, tunnel=True)

    assert url == "http://127.0.0.1:2024/ok"


def test_readiness_url_polls_the_server_url_when_direct():
    url = readiness_url("http://127.0.0.1:8123", port=8123, tunnel=False)

    assert url == "http://127.0.0.1:8123/ok"


def test_start_studio_probes_locally_while_tunneling():
    fake = FakeRuntime(modules=("google.colab",), api_url="https://x.trycloudflare.com")

    session = start_studio(runtime=fake.build())

    assert session.api_url == "https://x.trycloudflare.com"
