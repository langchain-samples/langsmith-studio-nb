"""Detect where the kernel is running and whether Studio can reach it directly."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Container, Mapping

_KAGGLE_VARIABLES = ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE")
_BINDER_VARIABLES = ("BINDER_SERVICE_HOST", "BINDER_REQUEST", "BINDER_REPO_URL")
_JUPYTERHUB_VARIABLES = ("JUPYTERHUB_USER", "JUPYTERHUB_API_TOKEN")


class NotebookEnvironment(StrEnum):
    """The notebook host a kernel is running under."""

    COLAB = "colab"
    KAGGLE = "kaggle"
    BINDER = "binder"
    JUPYTERHUB = "jupyterhub"
    LOCAL = "local"


_REMOTE = frozenset(
    {
        NotebookEnvironment.COLAB,
        NotebookEnvironment.KAGGLE,
        NotebookEnvironment.BINDER,
        NotebookEnvironment.JUPYTERHUB,
    }
)


def detect_environment(
    *, modules: Container[str], environ: Mapping[str, str]
) -> NotebookEnvironment:
    """Identify the notebook host from imported modules and environment variables."""
    if "google.colab" in modules:
        return NotebookEnvironment.COLAB
    if any(variable in environ for variable in _KAGGLE_VARIABLES):
        return NotebookEnvironment.KAGGLE
    # Binder runs on JupyterHub and sets both sets of variables, so it must win.
    if any(variable in environ for variable in _BINDER_VARIABLES):
        return NotebookEnvironment.BINDER
    if any(variable in environ for variable in _JUPYTERHUB_VARIABLES):
        return NotebookEnvironment.JUPYTERHUB
    return NotebookEnvironment.LOCAL


def needs_tunnel(environment: NotebookEnvironment) -> bool:
    """Report whether the browser needs a public URL to reach this kernel.

    Studio runs in the browser, so it can only reach `localhost` when the kernel
    shares a machine with it.
    """
    return environment in _REMOTE
