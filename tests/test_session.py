import pytest

from langsmith_studio_nb import _session
from langsmith_studio_nb._session import (
    StudioSession,
    start_studio,
    stop_studio,
)
from tests.conftest import FakeRuntime, FakeTunnelProcess, FakeWorker


def test_start_studio_serves_the_notebook_agent_and_returns_the_studio_url():
    fake = FakeRuntime()

    session = start_studio(runtime=fake.build())

    assert session == StudioSession(
        api_url="http://127.0.0.1:2024",
        studio_url=(
            "https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024&organizationId=ws-1"
        ),
        tunnel=False,
        graphs=("agent",),
    )
    assert fake.rendered == [(session.studio_url, None)]
    assert fake.server_calls[0]["graphs"] == {"agent": "__main__:agent"}


def test_start_studio_tunnels_when_the_kernel_is_remote():
    fake = FakeRuntime(modules=("google.colab",))

    session = start_studio(runtime=fake.build())

    assert session.tunnel is True
    assert fake.server_calls[0]["tunnel"] is False
    assert fake.opened_tunnel_ports == [2024]


def test_start_studio_honors_an_explicit_tunnel_choice():
    fake = FakeRuntime(modules=("google.colab",))

    session = start_studio(runtime=fake.build(), tunnel=False)

    assert session.tunnel is False
    assert fake.server_calls[0]["tunnel"] is False


def test_start_studio_serves_a_named_variable_on_a_chosen_port():
    fake = FakeRuntime(namespace={"researcher": object()})

    start_studio("researcher", port=8123, runtime=fake.build())

    assert fake.server_calls[0]["graphs"] == {"researcher": "__main__:researcher"}
    assert fake.server_calls[0]["port"] == 8123


def test_start_studio_serves_several_agents_at_once():
    fake = FakeRuntime(namespace={"planner": object(), "writer": object()})

    session = start_studio("planner", "writer", runtime=fake.build())

    assert fake.server_calls[0]["graphs"] == {
        "planner": "__main__:planner",
        "writer": "__main__:writer",
    }
    assert session.graphs == ("planner", "writer")


def test_start_studio_rejects_one_undefined_variable_among_several():
    fake = FakeRuntime(namespace={"planner": object()})

    with pytest.raises(NameError, match="'writer'"):
        start_studio("planner", "writer", runtime=fake.build())

    assert fake.server_calls == []


def test_start_studio_moves_off_a_port_that_is_taken():
    fake = FakeRuntime(port_is_free=False, free_port=51234)

    start_studio(runtime=fake.build())

    assert fake.server_calls[0]["port"] == 51234


def test_start_studio_reports_a_tunnel_that_never_came_up():
    fake = FakeRuntime(modules=("google.colab",), tunnel_error=TimeoutError())

    with pytest.raises(RuntimeError, match="re-run this cell"):
        start_studio(runtime=fake.build())

    assert fake.rendered == []
    assert _session._state is None
    assert fake.worker.stop_calls == 1
    assert fake.logging_restored == 1


def test_spawn_failure_restores_logging():
    fake = FakeRuntime(spawn_error=RuntimeError("thread unavailable"))

    with pytest.raises(RuntimeError, match="thread unavailable"):
        start_studio(runtime=fake.build())

    assert fake.logging_restored == 1
    assert _session._state is None


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

    assert _session._state is None


def test_start_studio_times_out_when_the_url_never_appears():
    fake = FakeRuntime(probes=[], tick=0.3)

    with pytest.raises(TimeoutError, match="did not answer within 1s"):
        start_studio(runtime=fake.build(), timeout=1.0)

    assert fake.sleeps == [0.5, 0.5, 0.5]
    assert fake.worker.stop_calls == 1
    assert _session._state is None


def test_startup_error_records_when_the_worker_cannot_be_cleaned_up():
    fake = FakeRuntime(worker=FakeWorker(stops=False), probes=[], tick=0.3)

    with pytest.raises(TimeoutError) as raised:
        start_studio(runtime=fake.build(), timeout=1.0)

    assert raised.value.__notes__ == [
        "The previous agent server did not stop within 20s, so a replacement was not started."
    ]
    assert _session._state is not None


def test_start_studio_restarts_a_running_server():
    fake = FakeRuntime()
    runtime = fake.build()

    start_studio(runtime=runtime)
    start_studio(runtime=runtime)

    assert fake.worker.joins == [20.0]
    assert len(fake.server_calls) == 2


def test_stop_studio_stops_the_owned_worker_and_clears_the_state():
    fake = FakeRuntime()
    runtime = fake.build()
    start_studio(runtime=runtime)

    stop_studio(runtime=runtime)

    assert fake.worker.stop_calls == 1
    assert _session._state is None


def test_start_and_stop_preserve_an_existing_server_url():
    fake = FakeRuntime(environ={"LANGGRAPH_API_URL": "https://existing.example"})
    runtime = fake.build()

    start_studio(runtime=runtime)
    stop_studio(runtime=runtime)

    assert fake.environ["LANGGRAPH_API_URL"] == "https://existing.example"


def test_stop_studio_is_safe_with_no_server_running():
    stop_studio()

    assert _session._state is None


def test_stop_studio_leaves_unowned_tunnels_alone():
    fake = FakeRuntime(modules=("google.colab",))
    runtime = fake.build()
    unowned = FakeTunnelProcess()
    start_studio(runtime=runtime)

    stop_studio(runtime=runtime)

    assert fake.tunnels[0].killed is True
    assert unowned.killed is False


def test_restart_refuses_to_overlap_a_worker_that_did_not_stop():
    fake = FakeRuntime(worker=FakeWorker(stops=False))
    runtime = fake.build()
    start_studio(runtime=runtime)

    with pytest.raises(RuntimeError, match="replacement was not started"):
        start_studio(runtime=runtime)

    assert len(fake.server_calls) == 1
    assert _session._state is not None


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


def test_verbose_restart_restores_levels_from_the_quiet_session():
    fake = FakeRuntime()
    runtime = fake.build()
    start_studio(runtime=runtime)

    start_studio(runtime=runtime, verbose=True)

    assert fake.quieted == 1
    assert fake.logging_restored == 1
    assert fake.server_calls[1]["server_level"] == "INFO"


def test_start_studio_hints_at_allowed_domains_when_tunneling():
    fake = FakeRuntime(modules=("google.colab",))

    start_studio(runtime=fake.build())

    _, hint = fake.rendered[0]
    assert hint is not None
    assert "*.trycloudflare.com" in hint


def test_session_renders_as_nothing_when_echoed():
    """The link is already displayed; the dataclass repr would duplicate it."""
    session = StudioSession(
        api_url="http://x", studio_url="http://y", tunnel=False, graphs=("agent",)
    )

    assert session._repr_html_() == ""


def test_start_studio_probes_locally_while_tunneling():
    fake = FakeRuntime(modules=("google.colab",))

    session = start_studio(runtime=fake.build())

    assert session.api_url == "https://x.trycloudflare.com"
    assert fake.probed[0] == "http://127.0.0.1:2024/ok"


def _tunneling_runtime():
    return FakeRuntime(modules=("google.colab",))


def test_start_studio_keeps_the_tunnel_it_is_already_running():
    """A fresh quick tunnel per restart is what Cloudflare rate limits."""
    fake = _tunneling_runtime()
    runtime = fake.build()

    start_studio(runtime=runtime)
    session = start_studio("agent", runtime=runtime)

    assert fake.server_calls[1]["tunnel"] is False
    assert session.api_url == "https://x.trycloudflare.com"
    assert session.tunnel is True
    assert [tunnel.killed for tunnel in fake.tunnels] == [False]


def test_restart_failure_releases_the_tunnel_kept_during_transition():
    fake = _tunneling_runtime()
    runtime = fake.build()
    start_studio(runtime=runtime)
    fake.spawn_error = RuntimeError("thread unavailable")

    with pytest.raises(RuntimeError, match="thread unavailable"):
        start_studio(runtime=runtime)

    assert fake.tunnels[0].killed is True
    assert _session._state is None


def test_start_studio_opens_a_new_tunnel_when_the_old_one_stopped_answering():
    """cloudflared outlives the tunnel Cloudflare dropped, so only the URL can be trusted."""
    fake = _tunneling_runtime()
    runtime = fake.build()

    start_studio(runtime=runtime)
    fake.unreachable = ("https://x.trycloudflare.com",)
    start_studio(runtime=runtime)

    assert fake.server_calls[1]["tunnel"] is False
    assert len(fake.tunnels) == 2
    assert fake.tunnels[0].killed is True


def test_start_studio_asks_the_tunnel_itself_before_keeping_it():
    fake = _tunneling_runtime()
    runtime = fake.build()

    start_studio(runtime=runtime)
    fake.probed.clear()
    start_studio(runtime=runtime)

    assert fake.probed[0] == "https://x.trycloudflare.com/ok"


def test_start_studio_drops_a_tunnel_whose_port_was_taken():
    """The tunnel forwards to one port only, so a server that moves outruns it."""
    fake = _tunneling_runtime()
    runtime = fake.build()

    start_studio(runtime=runtime)
    fake.port_is_free_value = False
    start_studio(runtime=runtime)

    assert fake.server_calls[1]["tunnel"] is False
    assert fake.server_calls[1]["port"] == 51234
    assert fake.tunnels[0].killed is True


def test_stop_studio_gives_up_the_tunnel():
    fake = _tunneling_runtime()
    runtime = fake.build()

    start_studio(runtime=runtime)
    stop_studio(runtime=runtime)
    start_studio(runtime=runtime)

    assert fake.tunnels[0].killed is True
    assert len(fake.tunnels) == 2


def test_start_studio_remembers_no_tunnel_when_serving_directly():
    fake = FakeRuntime()

    start_studio(runtime=fake.build(), tunnel=False)

    assert _session._state is not None
    assert _session._state.tunnel is None
