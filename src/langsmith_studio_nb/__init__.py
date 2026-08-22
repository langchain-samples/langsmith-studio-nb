"""Open LangGraph Studio on an agent defined in your notebook.

Works on Colab, where it tunnels, and in your own Jupyter, where it does not:

    from langsmith_studio_nb import start_studio

    start_studio()
"""

from __future__ import annotations

from importlib.metadata import version

from langsmith_studio_nb._runtime import OpenedTunnel, Runtime
from langsmith_studio_nb._session import StudioSession, start_studio, stop_studio
from langsmith_studio_nb._urls import STUDIO_ORIGIN, studio_url

__version__ = version("langsmith-studio-nb")

__all__ = [
    "STUDIO_ORIGIN",
    "OpenedTunnel",
    "Runtime",
    "StudioSession",
    "__version__",
    "start_studio",
    "stop_studio",
    "studio_url",
]
