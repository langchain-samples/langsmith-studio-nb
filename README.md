# langsmith-studio-nb

Open [LangGraph Studio](https://docs.langchain.com/langsmith/studio) on an agent you defined in a notebook cell.

Studio runs in your browser and connects to an agent server over HTTP. In a notebook that server has to be started from Python, on a background thread, and — when the kernel lives on a different machine than the browser, as in Colab — reached through a public URL. `langsmith-studio-nb` does all of that in one call:

```python
from langsmith_studio_nb import start_studio

start_studio()
```

You get a plain-text Studio link and nothing else — the server's own logs are silenced, since a notebook cell stops scrolling the moment it finishes executing. Pass `verbose=True` when you need them.

## Install

```python
%pip install git+https://github.com/langchain-samples/langsmith-studio-nb.git
```

Requires Python 3.11+. `langgraph-cli[inmem]` is installed as a dependency, so this is the only install line you need.

## Quickstart

```python
%pip install -q git+https://github.com/langchain-samples/langsmith-studio-nb.git deepagents
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
from langsmith_studio_nb import start_studio

start_studio()
```

Edit the agent and re-run the last two cells to pick up your changes — `start_studio` stops the previous server first.

## Several agents at once

Pass every variable you want to demo and pick between them in Studio's graph menu:

```python
start_studio("planner", "writer")
```

One server, one tunnel, one link — and Studio switches between the graphs without a restart.

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

### `start_studio(*variables, port=2024, tunnel=None, timeout=180, verbose=False)`

Serves the compiled graphs bound to `variables` in the notebook namespace — `agent` by default — and displays a Studio link. Each variable becomes a graph of the same name. Returns a `StudioSession` with `api_url`, `studio_url`, `tunnel`, and `graphs`.

Pass `verbose=True` to see the agent server's logs. Raises `NameError` if a variable is undefined, `RuntimeError` if the server dies during startup or the tunnel never comes up, and `TimeoutError` if the server never answers. A busy port is replaced with a free one automatically.

### `stop_studio()`

Stops the running server and its tunnel. Safe to call when nothing is running. Use it to force the next `start_studio` onto a fresh tunnel.

### `Runtime`

Every side effect the package performs, in one injectable frozen dataclass. Substitute it in tests:

```python
from langsmith_studio_nb import Runtime, start_studio

start_studio(runtime=Runtime(probe=lambda url: True, ...))
```

## Notes

- **No IPython dependency.** The package uses whatever IPython your notebook already has, so installing it will not upgrade Colab's pinned `ipython==7.34.0` out from under `google-colab`.
- **Studio must trust the tunnel domain.** On first connect, Studio blocks unknown hosts. Add `*.trycloudflare.com` under Advanced Settings → Allowed Domains — the wildcard, not the exact host Studio offers to add for you, which changes every time a tunnel opens. The list lives in browser localStorage, so it is per user and per browser; there is no workspace-level setting. `start_studio` prints this reminder whenever it tunnels.
- **Restarting keeps the tunnel.** Cloudflare rate limits quick tunnels per IP address (`error code: 1015`, `429 Too Many Requests`), and notebook hosts share theirs — a demo that restarts a few times can otherwise stop getting tunnels for a while. So a restart on the same port reuses the `cloudflared` already running: the link does not change, and the restart takes about a second instead of five. `stop_studio()` gives the tunnel up; the next `start_studio()` opens a fresh one.
- **A failed tunnel is an error, not a link.** When `cloudflared` does not report a URL, the server falls back to a URL only the kernel can reach; `start_studio` stops it and raises rather than printing a link that can only fail to fetch.
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
