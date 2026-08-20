"""Boot a real agent server. Deselected by default; run with `-m integration`."""

from __future__ import annotations

import json
import sys
import urllib.request
from dataclasses import dataclass

import pytest
from langgraph.graph import END, START, StateGraph

from langsmith_studio_nb import start_studio, stop_studio

pytestmark = pytest.mark.integration


@dataclass
class State:
    x: int


def _tiny_graph():
    builder = StateGraph(State)
    builder.add_node("bump", lambda state: {"x": state.x + 1})
    builder.add_edge(START, "bump")
    builder.add_edge("bump", END)
    return builder.compile()


@pytest.fixture
def notebook_agent():
    namespace = sys.modules["__main__"].__dict__
    namespace["agent"] = _tiny_graph()
    namespace["second_agent"] = _tiny_graph()
    yield
    namespace.pop("agent", None)
    namespace.pop("second_agent", None)


def _assistants(api_url: str) -> list[str]:
    request = urllib.request.Request(
        f"{api_url}/assistants/search",
        data=json.dumps({"limit": 5}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return sorted(assistant["graph_id"] for assistant in json.load(response))


def test_serves_the_agent_and_shuts_down(notebook_agent, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the server writes into the working directory

    session = start_studio(tunnel=False, timeout=120)
    try:
        assistants = _assistants(session.api_url)
    finally:
        stop_studio()

    assert assistants == ["agent"]
    assert session.studio_url.startswith("https://smith.langchain.com/studio/?baseUrl=")


def test_serves_two_agents_at_once(notebook_agent, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    session = start_studio("agent", "second_agent", tunnel=False, timeout=120)
    try:
        assistants = _assistants(session.api_url)
    finally:
        stop_studio()

    assert assistants == ["agent", "second_agent"]


def test_restarting_serves_the_new_graph(notebook_agent, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    start_studio(tunnel=False, timeout=120)
    session = start_studio("second_agent", tunnel=False, timeout=120)
    try:
        assistants = _assistants(session.api_url)
    finally:
        stop_studio()

    assert "second_agent" in assistants
