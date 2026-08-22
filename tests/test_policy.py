import pytest

from langsmith_studio_nb._policy import (
    DEFAULT_GRAPH_NAME,
    TUNNEL_PROTOCOLS,
    Request,
    missing_variables,
    plan,
    should_tunnel,
    tunnel_still_reaches,
)


def test_plan_defaults_to_the_agent_variable():
    assert plan((), port=2024, tunnel=False, namespace={"agent": object()}) == Request(
        graphs=(DEFAULT_GRAPH_NAME,), port=2024, tunnel=False
    )


def test_plan_keeps_the_order_the_caller_asked_for():
    namespace = {"writer": object(), "planner": object()}

    assert plan(("writer", "planner"), port=8000, tunnel=True, namespace=namespace).graphs == (
        "writer",
        "planner",
    )


def test_plan_names_the_first_variable_the_notebook_is_missing():
    with pytest.raises(NameError, match="'writer'"):
        plan(("planner", "writer"), port=2024, tunnel=False, namespace={"planner": object()})


@pytest.mark.parametrize(
    ("explicit", "in_colab", "expected"),
    [(None, True, True), (None, False, False), (True, False, True), (False, True, False)],
)
def test_should_tunnel(explicit, in_colab, expected):
    assert should_tunnel(explicit, in_colab=in_colab) is expected


def test_missing_variables_reports_them_all_in_order():
    assert missing_variables(("a", "b", "c"), {"b"}) == ("a", "c")


def test_tunnel_still_reaches_a_server_that_stayed_put():
    assert tunnel_still_reaches(tunnel_port=2024, bound_port=2024) is True
    assert tunnel_still_reaches(tunnel_port=2024, bound_port=51234) is False


def test_every_tunnel_attempt_tries_a_route_the_last_one_did_not():
    """Each one spends a quick tunnel against a rate limit the whole host shares."""
    assert len(TUNNEL_PROTOCOLS) == len(set(TUNNEL_PROTOCOLS))
    assert TUNNEL_PROTOCOLS[0] is None  # cloudflared's own default, over UDP
    assert "http2" in TUNNEL_PROTOCOLS  # the same port over TCP
