import os
import sys
import types

import pytest

from langsmith_studio_nb._runtime import Runtime
from langsmith_studio_nb._secrets import load_secret


def _install_module(monkeypatch, name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def test_load_secret_prefers_the_environment():
    environ = {"LANGSMITH_API_KEY": "from-environ"}
    runtime = Runtime(environ=environ, secret=lambda name: "from-host-store")

    assert load_secret("LANGSMITH_API_KEY", runtime=runtime) == "from-environ"


def test_load_secret_falls_back_to_the_host_store():
    environ = {}
    runtime = Runtime(environ=environ, secret=lambda name: f"host-{name}")

    assert load_secret("LANGSMITH_API_KEY", runtime=runtime) == "host-LANGSMITH_API_KEY"
    assert environ["LANGSMITH_API_KEY"] == "host-LANGSMITH_API_KEY"


def test_load_secret_raises_when_nothing_has_it():
    runtime = Runtime(environ={}, secret=lambda name: None)

    with pytest.raises(RuntimeError, match="LANGSMITH_API_KEY is not set"):
        load_secret("LANGSMITH_API_KEY", runtime=runtime)


def test_load_secret_can_be_optional():
    environ = {}
    runtime = Runtime(environ=environ, secret=lambda name: None)

    assert load_secret("LANGSMITH_API_KEY", required=False, runtime=runtime) is None
    assert environ == {}


def test_load_secret_treats_an_empty_value_as_absent():
    runtime = Runtime(environ={"LANGSMITH_API_KEY": ""}, secret=lambda name: None)

    with pytest.raises(RuntimeError):
        load_secret("LANGSMITH_API_KEY", runtime=runtime)


def test_load_secret_uses_the_real_runtime_by_default(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "from-real-environ")

    assert load_secret("LANGSMITH_API_KEY") == "from-real-environ"


@pytest.fixture
def unset_key(monkeypatch):
    """Yield a variable name absent from the environment, and absent again afterwards.

    `load_secret` writes to the real `os.environ` in these tests, which monkeypatch
    does not know to undo.
    """
    name = "STUDIO_NB_TEST_KEY"
    monkeypatch.delenv(name, raising=False)
    yield name
    monkeypatch.delenv(name, raising=False)


def test_load_secret_reaches_colab_secrets_through_the_real_runtime(monkeypatch, unset_key):
    """End to end, with no injected Runtime, so the host dispatch has to work."""

    class UserData:
        @staticmethod
        def get(name):
            return f"colab-{name}"

    _install_module(monkeypatch, "google.colab", userdata=UserData)

    assert load_secret(unset_key) == f"colab-{unset_key}"
    assert os.environ[unset_key] == f"colab-{unset_key}"


def test_load_secret_reaches_kaggle_secrets_through_the_real_runtime(monkeypatch, unset_key):
    class UserSecretsClient:
        def get_secret(self, name):
            return f"kaggle-{name}"

    _install_module(monkeypatch, "kaggle_secrets", UserSecretsClient=UserSecretsClient)
    monkeypatch.setenv("KAGGLE_URL_BASE", "https://kaggle.com")

    assert load_secret(unset_key) == f"kaggle-{unset_key}"
    assert os.environ[unset_key] == f"kaggle-{unset_key}"


def test_load_secret_message_names_every_host():
    runtime = Runtime(environ={}, secret=lambda name: None)

    with pytest.raises(RuntimeError) as caught:
        load_secret("ANTHROPIC_API_KEY", runtime=runtime)

    message = str(caught.value)
    assert "Colab" in message
    assert "Kaggle" in message
    assert "load_dotenv()" in message
