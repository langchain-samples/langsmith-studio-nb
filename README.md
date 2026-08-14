# nbstudio

Open [LangGraph Studio](https://docs.langchain.com/langsmith/studio) on an agent you defined in a notebook cell.

Studio runs in your browser and connects to an agent server over HTTP. In a notebook that server has to be started from Python, on a background thread, and — when the kernel lives on a different machine than the browser, as in Colab — reached through a public URL. `nbstudio` does all of that in one call:

```python
from nbstudio import start_studio

start_studio()
```

You get a clickable Studio link. That's the whole API.

## Install

```python
%pip install git+https://github.com/langchain-samples/nbstudio.git
```

Requires Python 3.11+. `langgraph-cli[inmem]` is installed as a dependency, so this is the only install line you need.

## Quickstart

```python
%pip install -q git+https://github.com/langchain-samples/nbstudio.git deepagents
```

```python
import getpass, os

os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Anthropic API key: ")
os.environ["LANGSMITH_API_KEY"] = getpass.getpass("LangSmith API key: ")
os.environ["LANGSMITH_TRACING"] = "true"
```

```python
from deepagents import create_deep_agent


def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"It's always sunny in {city}!"


agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
    tools=[get_weather],
    system_prompt="You are a helpful research assistant.",
)
```

```python
from nbstudio import start_studio

start_studio()
```

Edit the agent and re-run the last two cells to pick up your changes — `start_studio` stops the previous server first.

## Where it works

`start_studio` tunnels only when the browser cannot reach the kernel directly.

| Environment | Detected by | Tunnel |
| --- | --- | --- |
| Google Colab | `google.colab` imported | yes |
| Kaggle | `KAGGLE_*` variables | yes |
| Binder | `BINDER_*` variables | yes |
| JupyterHub | `JUPYTERHUB_*` variables | yes |
| Local Jupyter, JupyterLab, VS Code | default | no |

Tunnels use a Cloudflare quick tunnel. The `cloudflared` binary downloads automatically on first use.

Override the choice when the guess is wrong:

```python
start_studio(tunnel=True)  # e.g. a browser that blocks https -> localhost
start_studio(tunnel=False)  # e.g. a remote kernel you already expose yourself
```

## API

### `start_studio(variable="agent", *, port=2024, tunnel=None, timeout=180)`

Serves the compiled graph bound to `variable` in the notebook namespace and displays a Studio link. Returns a `StudioSession` with `api_url`, `studio_url`, and `tunnel`.

Raises `NameError` if the variable is undefined, `RuntimeError` if the server dies during startup, and `TimeoutError` if it never answers. A busy port is replaced with a free one automatically.

### `stop_studio()`

Stops the running server and its tunnel. Safe to call when nothing is running.

### `Runtime`

Every side effect the package performs, in one injectable frozen dataclass. Substitute it in tests:

```python
from nbstudio import Runtime, start_studio

start_studio(runtime=Runtime(probe=lambda url: True, ...))
```

## Notes

- **A tunnel URL is public and unauthenticated** for as long as the cell runs. Fine for a demo agent; think twice with anything sensitive. The tunnel dies with the kernel.
- **Edits need a restart.** The graph object is registered when the server boots, so re-run `start_studio()` after changing your agent. Hot reload is not available from a notebook.
- **Colab drops idle runtimes** after roughly 90 minutes. You get a new tunnel URL after reconnecting.
- **Kaggle needs internet enabled** in the notebook settings, otherwise the tunnel cannot start.

## Development

```bash
uv sync
uv run ruff format . && uv run ruff check . && uv run ty check
uv run pytest                    # unit tests, 100% coverage enforced
uv run pytest -m integration     # boots a real agent server
```
