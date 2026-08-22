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
    fake = FakeRuntime(statuses=[None, 200])

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
    fake = FakeRuntime(statuses=[], tick=0.3)

    with pytest.raises(TimeoutError, match="did not answer within 1s"):
        start_studio(runtime=fake.build(), timeout=1.0)

    assert fake.sleeps == [0.5, 0.5, 0.5]
    assert fake.worker.stop_calls == 1
    assert _session._state is None


def test_startup_error_releases_resources_even_when_the_worker_will_not_stop():
    """A server that outlives the join must not keep the tunnel and log levels with it."""
    fake = FakeRuntime(
        modules=("google.colab",), worker=FakeWorker(stops=False), statuses=[], tick=0.3
    )

    with pytest.raises(TimeoutError, match="did not answer within 1s"):
        start_studio(runtime=fake.build(), timeout=1.0)

    assert _session._state is None
    assert fake.logging_restored == 1


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


def test_start_studio_ignores_a_server_url_it_did_not_publish():
    """A leftover value would point this session at an endpoint it does not own."""
    fake = FakeRuntime(environ={"LANGGRAPH_API_URL": "https://stale.example"})

    session = start_studio(runtime=fake.build())

    assert session.api_url == "http://127.0.0.1:2024"


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


def test_restart_moves_off_a_worker_that_did_not_stop():
    """Refusing to start would leave the notebook with no server at all."""
    fake = FakeRuntime(worker=FakeWorker(stops=False), port_is_free=False)
    runtime = fake.build()
    start_studio(runtime=runtime)

    session = start_studio(runtime=runtime)

    assert fake.server_calls[1]["port"] == 51234
    assert session.api_url == "http://127.0.0.1:51234"


def test_restart_keeps_the_tunnel_while_the_old_server_drains():
    """A draining server gives up its port, so its tunnel is still worth keeping."""
    fake = FakeRuntime(modules=("google.colab",), worker=FakeWorker(stops=False))
    runtime = fake.build()
    start_studio(runtime=runtime)

    session = start_studio(runtime=runtime)

    assert len(fake.tunnels) == 1
    assert fake.tunnels[0].killed is False
    assert session.api_url == "https://x.trycloudflare.com"


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
    """The package already displayed the link, so the repr would duplicate it."""
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
    """cloudflared outlives the tunnel Cloudflare dropped, so only the URL is worth trusting."""
    fake = _tunneling_runtime()
    runtime = fake.build()

    fake.tunnel_urls = ["https://dropped.trycloudflare.com", "https://fresh.trycloudflare.com"]
    start_studio(runtime=runtime)
    fake.drop_tunnel(0)
    session = start_studio(runtime=runtime)

    assert fake.server_calls[1]["tunnel"] is False
    assert len(fake.tunnels) == 2
    assert fake.tunnels[0].killed is True
    assert session.api_url == "https://fresh.trycloudflare.com"


def test_start_studio_asks_cloudflared_before_keeping_its_tunnel():
    fake = _tunneling_runtime()
    runtime = fake.build()

    start_studio(runtime=runtime)
    fake.probed.clear()
    start_studio(runtime=runtime)

    assert fake.probed[0] == fake.ready_urls[0]


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


def test_start_studio_replaces_a_tunnel_that_never_connects():
    """cloudflared hands out the URL before it reaches Cloudflare, and may never reach it."""
    fake = FakeRuntime(
        modules=("google.colab",),
        tunnel_urls=["https://dead.trycloudflare.com", "https://live.trycloudflare.com"],
        tunnel_ready=[False, True],
        tick=3.0,
    )

    session = start_studio(runtime=fake.build())

    assert session.api_url == "https://live.trycloudflare.com"
    assert [tunnel.killed for tunnel in fake.tunnels] == [True, False]
    assert fake.opened_tunnel_ports == [2024, 2024]


def test_start_studio_falls_back_to_http2_after_the_default_protocol_fails():
    """The default rides UDP, which notebook hosts drop; http2 rides TCP 443."""
    fake = FakeRuntime(modules=("google.colab",), tunnel_ready=[False, True], tick=3.0)

    start_studio(runtime=fake.build())

    assert fake.opened_tunnel_protocols == [None, "http2"]


def test_start_studio_gives_up_after_three_tunnels_that_never_connect():
    """Better an error than a link that cannot work; each attempt costs a tunnel."""
    fake = FakeRuntime(
        modules=("google.colab",),
        tunnel_ready=[False, False, False],
        tick=3.0,
    )

    with pytest.raises(RuntimeError, match="never reached Cloudflare"):
        start_studio(runtime=fake.build())

    assert len(fake.tunnels) == 3
    assert all(tunnel.killed for tunnel in fake.tunnels)
    assert fake.rendered == []
    assert _session._state is None
    assert fake.logging_restored == 1


def test_start_studio_waits_between_tunnel_attempts():
    fake = FakeRuntime(
        modules=("google.colab",),
        tunnel_ready=[False, False, False],
        tick=3.0,
    )

    with pytest.raises(RuntimeError):
        start_studio(runtime=fake.build())

    assert fake.sleeps.count(2.0) == 2  # paused between attempts, not after the last


def test_start_studio_reports_what_stopped_the_tunnel():
    """Not every failure is rate limiting; Kaggle without internet cannot download cloudflared."""
    fake = FakeRuntime(
        modules=("google.colab",), tunnel_error=OSError("cloudflared download failed")
    )

    with pytest.raises(RuntimeError, match="could not be opened: cloudflared download failed"):
        start_studio(runtime=fake.build())

    assert _session._state is None


def test_start_studio_tunnels_to_the_port_the_server_reports():
    """The server resolves the port again itself and moves off a busy one silently."""
    fake = FakeRuntime(modules=("google.colab",), api_url="http://127.0.0.1:51999")

    start_studio(runtime=fake.build())

    assert fake.server_calls[0]["port"] == 2024
    assert fake.opened_tunnel_ports == [51999]


def test_start_studio_falls_back_to_the_port_it_asked_for():
    fake = FakeRuntime(modules=("google.colab",), api_url="http://127.0.0.1")

    start_studio(runtime=fake.build())

    assert fake.opened_tunnel_ports == [2024]


def test_restart_replaces_a_tunnel_cloudflare_dropped():
    """cloudflared keeps running and keeps its URL, connected to nothing."""
    fake = FakeRuntime(
        modules=("google.colab",),
        tunnel_urls=["https://first.trycloudflare.com", "https://second.trycloudflare.com"],
        tick=3.0,
    )
    runtime = fake.build()
    start_studio(runtime=runtime)
    fake.drop_tunnel(0)

    session = start_studio(runtime=runtime)

    assert fake.tunnels[0].killed is True
    assert session.api_url == "https://second.trycloudflare.com"
