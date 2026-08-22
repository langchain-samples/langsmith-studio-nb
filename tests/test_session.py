import threading
from dataclasses import replace

import pytest

from langsmith_studio_nb import _session
from langsmith_studio_nb._session import (
    StudioSession,
    start_studio,
    stop_studio,
)
from tests.conftest import FakeRuntime, FakeTunnel, FakeWorker


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
    fake = FakeRuntime(in_colab=True)

    session = start_studio(runtime=fake.build())

    assert session.tunnel is True
    assert fake.server_calls[0]["tunnel"] is False
    assert fake.opened_tunnel_ports == [2024]


def test_start_studio_honors_an_explicit_tunnel_choice():
    fake = FakeRuntime(in_colab=True)

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
    fake = FakeRuntime(in_colab=True, tunnel_error=TimeoutError())

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
    fake = FakeRuntime(in_colab=True, worker=FakeWorker(stops=False), statuses=[], tick=0.3)

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

    stop_studio()

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
    fake = FakeRuntime(in_colab=True)
    runtime = fake.build()
    unowned = FakeTunnel()
    start_studio(runtime=runtime)

    stop_studio()

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
    fake = FakeRuntime(in_colab=True, worker=FakeWorker(stops=False))
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
    fake = FakeRuntime(in_colab=True)

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
    fake = FakeRuntime(in_colab=True)

    session = start_studio(runtime=fake.build())

    assert session.api_url == "https://x.trycloudflare.com"
    assert fake.probed[0] == "http://127.0.0.1:2024/ok"


def _tunneling_runtime():
    return FakeRuntime(in_colab=True)


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
    """cloudflared outlives the tunnel Cloudflare dropped, so only /ready is worth asking."""
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
    stop_studio()
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
        in_colab=True,
        tunnel_urls=["https://dead.trycloudflare.com", "https://live.trycloudflare.com"],
        tunnel_ready=[False, True],
        tick=3.0,
    )

    session = start_studio(runtime=fake.build())

    assert session.api_url == "https://live.trycloudflare.com"
    assert [tunnel.killed for tunnel in fake.tunnels] == [True, False]
    assert fake.opened_tunnel_ports == [2024, 2024]


def test_start_studio_falls_back_to_http2_after_the_default_protocol_fails():
    """The default rides UDP, which notebook hosts drop; http2 rides TCP 7844."""
    fake = FakeRuntime(in_colab=True, tunnel_ready=[False, True], tick=3.0)

    start_studio(runtime=fake.build())

    assert fake.opened_tunnel_protocols == [None, "http2"]


def test_start_studio_spends_one_tunnel_per_route_and_no_more():
    """Better an error than a link that cannot work, and a repeat buys no new route."""
    fake = FakeRuntime(in_colab=True, tunnel_ready=[False, False], tick=3.0)

    with pytest.raises(RuntimeError, match="never reached Cloudflare"):
        start_studio(runtime=fake.build())

    assert fake.opened_tunnel_protocols == [None, "http2"]
    assert all(tunnel.killed for tunnel in fake.tunnels)
    assert fake.rendered == []
    assert _session._state is None
    assert fake.logging_restored == 1


def test_start_studio_waits_between_tunnel_attempts():
    fake = FakeRuntime(in_colab=True, tunnel_ready=[False, False], tick=3.0)

    with pytest.raises(RuntimeError):
        start_studio(runtime=fake.build())

    assert fake.sleeps.count(2.0) == 1  # paused between attempts, not after the last


def test_start_studio_reports_what_stopped_the_tunnel():
    """Not every failure is rate limiting; a runtime with no internet cannot fetch cloudflared."""
    fake = FakeRuntime(in_colab=True, tunnel_error=OSError("cloudflared download failed"))

    with pytest.raises(RuntimeError, match="could not be opened: cloudflared download failed"):
        start_studio(runtime=fake.build())

    assert _session._state is None


def test_start_studio_tunnels_to_the_port_the_server_reports():
    """The server resolves the port again itself and moves off a busy one silently."""
    fake = FakeRuntime(in_colab=True, api_url="http://127.0.0.1:51999")

    start_studio(runtime=fake.build())

    assert fake.server_calls[0]["port"] == 2024
    assert fake.opened_tunnel_ports == [51999]


def test_start_studio_falls_back_to_the_port_it_asked_for():
    fake = FakeRuntime(in_colab=True, api_url="http://127.0.0.1")

    start_studio(runtime=fake.build())

    assert fake.opened_tunnel_ports == [2024]


def test_restart_replaces_a_tunnel_cloudflare_dropped():
    """cloudflared keeps running and keeps its URL, connected to nothing."""
    fake = FakeRuntime(
        in_colab=True,
        tunnel_urls=["https://first.trycloudflare.com", "https://second.trycloudflare.com"],
        tick=3.0,
    )
    runtime = fake.build()
    start_studio(runtime=runtime)
    fake.drop_tunnel(0)

    session = start_studio(runtime=runtime)

    assert fake.tunnels[0].killed is True
    assert session.api_url == "https://second.trycloudflare.com"


def test_start_studio_closes_a_tunnel_it_was_interrupted_while_checking():
    """Nothing owns the process yet, so a Ctrl-C here would strand it."""
    fake = FakeRuntime(in_colab=True)
    runtime = fake.build()
    fake.status_error = KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        start_studio(runtime=runtime)

    assert fake.tunnels[0].closes == 1
    assert _session._state is None


def test_start_studio_gives_the_api_url_variable_back_when_it_stops():
    """Other langgraph tooling in the notebook reads this; ours dies with the server."""
    fake = FakeRuntime(environ={"LANGGRAPH_API_URL": "http://example.invalid"})
    runtime = fake.build()

    start_studio(runtime=runtime)

    assert fake.environ["LANGGRAPH_API_URL"] == "http://127.0.0.1:2024"

    stop_studio()

    assert fake.environ["LANGGRAPH_API_URL"] == "http://example.invalid"


def test_stop_studio_leaves_no_dead_api_url_behind():
    fake = FakeRuntime()
    runtime = fake.build()
    start_studio(runtime=runtime)

    stop_studio()

    assert "LANGGRAPH_API_URL" not in fake.environ


def test_start_studio_replaces_a_tunnel_cloudflare_drops_mid_restart():
    """The kept tunnel was checked before the old server stopped; the link goes out later."""
    fake = FakeRuntime(
        in_colab=True,
        tunnel_urls=["https://first.trycloudflare.com", "https://second.trycloudflare.com"],
        tick=3.0,
    )
    runtime = fake.build()
    start_studio(runtime=runtime)

    def drop_it_once_the_reuse_check_has_passed(url):
        if url == fake.ready_urls[0]:
            fake.drop_tunnel(0)

    fake.after_probe = drop_it_once_the_reuse_check_has_passed
    session = start_studio(runtime=runtime)

    assert session.api_url == "https://second.trycloudflare.com"
    assert [tunnel.closes for tunnel in fake.tunnels] == [1, 0]


def test_stop_studio_releases_everything_when_the_worker_will_not_stop():
    fake = FakeRuntime(in_colab=True, worker=FakeWorker(stop_error=RuntimeError("wedged")))
    runtime = fake.build()
    start_studio(runtime=runtime)

    stop_studio()

    assert fake.tunnels[0].killed is True
    assert fake.logging_restored == 1
    assert _session._state is None


def test_stop_studio_restores_logging_when_the_tunnel_will_not_close():
    """Teardown has nothing to fall back on, so one broken step must not take the rest."""
    fake = FakeRuntime(in_colab=True)
    runtime = fake.build()
    start_studio(runtime=runtime)
    fake.tunnels[0].close_error = OSError("no such process")

    stop_studio()

    assert fake.logging_restored == 1
    assert "LANGGRAPH_API_URL" not in fake.environ


def test_stop_studio_gives_up_the_tunnel_when_the_join_is_interrupted():
    """A Ctrl-C on a slow shutdown would otherwise orphan cloudflared for the kernel's life."""
    fake = FakeRuntime(in_colab=True, worker=FakeWorker(join_error=KeyboardInterrupt()))
    runtime = fake.build()
    start_studio(runtime=runtime)

    with pytest.raises(KeyboardInterrupt):
        stop_studio()

    assert fake.tunnels[0].killed is True
    assert fake.logging_restored == 1
    assert _session._state is None


def test_a_stop_from_another_thread_waits_for_the_start_to_finish():
    """A stop landing mid-start would tear down the session `start_studio` is about to return."""
    fake = FakeRuntime()
    stopper = threading.Thread(target=stop_studio)

    def workspace_id():
        stopper.start()
        stopper.join(0.2)

        assert stopper.is_alive()  # waiting on the lock, not tearing this session down

        return "ws-1"

    runtime = replace(fake.build(), workspace_id=workspace_id)

    session = start_studio(runtime=runtime)
    stopper.join(5)

    assert session.api_url == "http://127.0.0.1:2024"
    assert _session._state is None  # the stop ran, but only once the start was done
    assert fake.tunnels == []


def test_a_second_start_waits_for_the_first_to_finish():
    """Two starts that interleave would each stop the other's server and tunnel."""
    fake = FakeRuntime(in_colab=True)
    second = threading.Thread(target=lambda: start_studio(runtime=fake.build()))

    def workspace_id():
        second.start()
        second.join(0.2)

        assert second.is_alive()  # waiting on the lock, not opening a tunnel of its own
        assert len(fake.tunnels) == 1

        return "ws-1"

    session = start_studio(runtime=replace(fake.build(), workspace_id=workspace_id))
    second.join(5)

    assert session.api_url == "https://x.trycloudflare.com"
    assert [tunnel.closes for tunnel in fake.tunnels] == [0]  # one tunnel, kept across both
    assert _session._state is not None


def test_restart_closes_the_tunnel_it_kept_when_teardown_is_interrupted():
    """A kept tunnel reaches its next owner before anything else is released."""
    fake = FakeRuntime(in_colab=True)
    runtime = fake.build()
    start_studio(runtime=runtime)
    fake.restore_error = KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        start_studio(runtime=runtime)

    assert fake.tunnels[0].closes == 1  # handed on, then closed by the caller that took it
    assert _session._state is None


def test_start_studio_closes_a_new_tunnel_when_rendering_the_link_is_interrupted():
    """A verified tunnel belongs to the session before anything downstream can fail."""
    fake = FakeRuntime(in_colab=True)

    def workspace_id():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        start_studio(runtime=replace(fake.build(), workspace_id=workspace_id))

    assert fake.tunnels[0].closes == 1
    assert fake.rendered == []
    assert _session._state is None


def test_start_studio_stops_a_worker_whose_thread_would_not_start():
    """The session owns the worker before it runs, so a failed start is still teardown's."""
    fake = FakeRuntime(worker=FakeWorker(start_error=RuntimeError("can't start new thread")))

    with pytest.raises(RuntimeError, match="can't start new thread"):
        start_studio(runtime=fake.build())

    assert fake.worker.stop_calls == 1  # found and stopped, not left running
    assert fake.logging_restored == 1
    assert "LANGGRAPH_API_URL" not in fake.environ
    assert _session._state is None
