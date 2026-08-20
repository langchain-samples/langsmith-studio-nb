"""Open LangGraph Studio on an agent defined in your notebook.

Works in Colab, Kaggle, Binder, JupyterHub, and local Jupyter or VS Code:

    from langsmith_studio_nb import start_studio

    start_studio()
"""

from __future__ import annotations

from importlib.metadata import version

from langsmith_studio_nb._environment import NotebookEnvironment, detect_environment, needs_tunnel
from langsmith_studio_nb._runtime import OpenedTunnel, Runtime
from langsmith_studio_nb._session import StudioSession, start_studio, stop_studio
from langsmith_studio_nb._urls import STUDIO_ORIGIN, studio_url

__version__ = version("langsmith-studio-nb")

__all__ = [
    "STUDIO_ORIGIN",
    "NotebookEnvironment",
    "OpenedTunnel",
    "Runtime",
    "StudioSession",
    "__version__",
    "detect_environment",
    "needs_tunnel",
    "start_studio",
    "stop_studio",
    "studio_url",
]
