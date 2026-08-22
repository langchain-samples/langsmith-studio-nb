"""Read an API key from wherever the current notebook host keeps its secrets."""

from __future__ import annotations

from langsmith_studio_nb._runtime import Runtime

MISSING = (
    "{name} is not set. On Colab, add it under \N{KEY} Secrets with 'Notebook access' "
    "turned on. On Kaggle, add it under Add-ons \N{RIGHTWARDS ARROW} Secrets and attach it "
    "to this notebook. Anywhere else, export it or put it in a .env file and call "
    "python-dotenv's load_dotenv() first."
)


def load_secret(name: str, *, required: bool = True, runtime: Runtime | None = None) -> str | None:
    """Put `name` into the environment and return it, from the host's secret store.

    Reads the environment first, so an exported variable wins over the host's store.
    So does one that `python-dotenv` already loaded from a `.env`. Writes into the
    environment because that is where the model and tracing SDKs go looking for a key.

    Args:
        name: Variable to resolve, such as `LANGSMITH_API_KEY`.
        required: Raise when the secret is missing, rather than returning None.
        runtime: Side effects to use. Substitute it in tests.

    Returns:
        The secret, or None when it is absent and `required` is False.

    Raises:
        RuntimeError: `required` is set and nothing this checks has the secret.
    """
    runtime = runtime or Runtime()
    value = runtime.environ.get(name) or runtime.secret(name)
    if not value:
        if required:
            raise RuntimeError(MISSING.format(name=name))
        return None
    runtime.environ[name] = value
    return value
