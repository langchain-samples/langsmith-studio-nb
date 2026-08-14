"""Open LangGraph Studio on an agent defined in your notebook.

Works in Colab, Kaggle, Binder, JupyterHub, and local Jupyter or VS Code:

    from nbstudio import start_studio

    start_studio()
"""

from __future__ import annotations

from importlib.metadata import version

from nbstudio._environment import NotebookEnvironment, detect_environment, needs_tunnel
from nbstudio._runtime import Runtime
from nbstudio._session import StudioSession, start_studio, stop_studio
from nbstudio._urls import STUDIO_ORIGIN, studio_url

__version__ = version("nbstudio")

__all__ = [
    "STUDIO_ORIGIN",
    "NotebookEnvironment",
    "Runtime",
    "StudioSession",
    "__version__",
    "detect_environment",
    "needs_tunnel",
    "start_studio",
    "stop_studio",
    "studio_url",
]
