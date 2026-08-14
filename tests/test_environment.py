import pytest

from langsmith_studio_nb._environment import NotebookEnvironment, detect_environment, needs_tunnel


@pytest.mark.parametrize(
    ("modules", "environ", "expected"),
    [
        (("google.colab",), {}, NotebookEnvironment.COLAB),
        ((), {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}, NotebookEnvironment.KAGGLE),
        ((), {"KAGGLE_URL_BASE": "https://kaggle.com"}, NotebookEnvironment.KAGGLE),
        ((), {"BINDER_SERVICE_HOST": "10.0.0.1"}, NotebookEnvironment.BINDER),
        ((), {"BINDER_REQUEST": "v2/gh/org/repo"}, NotebookEnvironment.BINDER),
        ((), {"BINDER_REPO_URL": "https://github.com/org/repo"}, NotebookEnvironment.BINDER),
        ((), {"JUPYTERHUB_USER": "dariel"}, NotebookEnvironment.JUPYTERHUB),
        ((), {"JUPYTERHUB_API_TOKEN": "token"}, NotebookEnvironment.JUPYTERHUB),
        ((), {}, NotebookEnvironment.LOCAL),
        (("ipykernel",), {"VSCODE_PID": "42"}, NotebookEnvironment.LOCAL),
    ],
)
def test_detect_environment(modules, environ, expected):
    assert detect_environment(modules=modules, environ=environ) == expected


def test_binder_wins_over_jupyterhub():
    environ = {"BINDER_SERVICE_HOST": "10.0.0.1", "JUPYTERHUB_USER": "jovyan"}

    assert detect_environment(modules=(), environ=environ) == NotebookEnvironment.BINDER


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        (NotebookEnvironment.COLAB, True),
        (NotebookEnvironment.KAGGLE, True),
        (NotebookEnvironment.BINDER, True),
        (NotebookEnvironment.JUPYTERHUB, True),
        (NotebookEnvironment.LOCAL, False),
    ],
)
def test_needs_tunnel(environment, expected):
    assert needs_tunnel(environment) is expected
