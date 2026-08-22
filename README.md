# langsmith-studio-nb

Open [LangGraph Studio](https://docs.langchain.com/langsmith/studio) on an agent you defined in a notebook cell.

Studio runs in your browser and connects to an agent server over HTTP. In a notebook you have to start that server yourself, from Python, on a background thread. When the kernel sits on a different machine than the browser, as it does on Colab, the server also needs a public URL. `langsmith-studio-nb` does all of that in one call:

```python
from langsmith_studio_nb import start_studio

start_studio()
```

You get a plain-text Studio link and nothing else. The package silences the server's own logs, because a notebook cell stops scrolling the moment it finishes executing. Pass `verbose=True` when you want them back.

## Install

```python
%pip install git+https://github.com/langchain-samples/langsmith-studio-nb.git
```

Requires Python 3.11+. `langgraph-cli[inmem]` comes with it, so this is the only install line you need.

## Quickstart

```python
%pip install -q git+https://github.com/langchain-samples/langsmith-studio-nb.git deepagents langchain-openai
```

```python
import os

from langsmith_studio_nb import load_secret

load_secret("OPENAI_API_KEY")  # Colab and Kaggle: from their secret stores
load_secret("LANGSMITH_API_KEY")  # elsewhere: from the environment
os.environ["LANGSMITH_TRACING"] = "true"
```

```python
from deepagents import create_deep_agent


def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"It's always sunny in {city}!"


agent = create_deep_agent(
    model="openai:gpt-5.6-luna",
    tools=[get_weather],
    system_prompt="You are a helpful research assistant.",
)
```

```python
from langsmith_studio_nb import start_studio

start_studio()
```

Edit the agent and re-run the last two cells to pick up your changes. `start_studio` stops the previous server before it starts the next one.

## Example notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-samples/langsmith-studio-nb/blob/main/examples/deep_agent_in_studio.ipynb)
[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/langchain-samples/langsmith-studio-nb/HEAD?labpath=examples%2Fdeep_agent_in_studio.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/langchain-samples/langsmith-studio-nb/blob/main/examples/deep_agent_in_studio.ipynb)

[**examples/deep_agent_in_studio.ipynb**](examples/deep_agent_in_studio.ipynb) is the quickstart above as a runnable notebook, plus the two things a hosted runtime adds: where to keep your API keys on each host, and the one Studio setting a tunnel needs. It runs top to bottom on Colab, Kaggle, Binder, or local Jupyter.

`load_secret` finds the keys. Locally and on Binder they come from a `.env` file, which you make with `cd examples && cp .env.example .env`. On Colab and Kaggle they come from the host's own store. The template sits beside the notebook, because the keys belong to the example rather than to the package.

Everything Binder needs lives in [`.binder/`](.binder), and all of it serves the example. The package declares its own dependencies in `pyproject.toml` and needs none of it. `requirements.txt` there adds the agent stack the notebook imports. `runtime.txt` pins the Python version that stack supports. `postBuild` lets JupyterLab see dotfiles, so a reader can create the `.env`. Each file says why in a comment. Keep the `.` line in `requirements.txt`: it is what installs this package from the commit Binder launched.

## Several agents at once

Pass every variable you want to demo and pick between them in Studio's graph menu:

```python
start_studio("planner", "writer")
```

One server, one tunnel, one link. Studio switches between the graphs without a restart.

## Where it works

`start_studio` tunnels only when the browser cannot reach the kernel directly.

| Environment | Detected by | Tunnel |
| --- | --- | --- |
| Google Colab | `google.colab` imported | yes |
| Kaggle | `KAGGLE_*` variables | yes |
| Binder | `BINDER_*` variables | yes |
| JupyterHub | `JUPYTERHUB_*` variables | yes |
| Local Jupyter, JupyterLab, VS Code | default | no |

Tunnels use a Cloudflare quick tunnel. The `cloudflared` binary downloads itself on first use.

Override the choice when the guess is wrong:

```python
start_studio(tunnel=True)  # e.g. a browser that blocks https -> localhost
start_studio(tunnel=False)  # e.g. a remote kernel you already expose yourself
```

## API

### `start_studio(*variables, port=2024, tunnel=None, timeout=180, verbose=False)`

Serves the compiled graphs bound to `variables` in the notebook namespace, `agent` by default, and displays a Studio link. Each variable becomes a graph of the same name. Returns a `StudioSession` carrying `api_url`, `studio_url`, `tunnel`, and `graphs`.

Pass `verbose=True` to see the agent server's logs. Raises `NameError` if a variable is undefined, `RuntimeError` if the server dies during startup or the tunnel never comes up, and `TimeoutError` if the server never answers. If the port you asked for is busy, it picks a free one instead.

### `stop_studio()`

Stops the running server and its tunnel, and puts the log levels back. Safe to call when nothing is running, and it never raises. A server draining a long request can outlive the wait, and it gives up the tunnel and restores the log levels either way. Use it to force the next `start_studio` onto a fresh tunnel.

### `load_secret(name, *, required=True)`

Puts `name` into `os.environ` and returns it, taken from whichever secret store the host has. It reads the environment first, so an exported variable wins over the store. So does one that `python-dotenv` has already loaded from a `.env`:

```python
from dotenv import load_dotenv

from langsmith_studio_nb import load_secret

load_dotenv()  # local and Binder: a .env file
load_secret("OPENAI_API_KEY")  # Colab: userdata. Kaggle: UserSecretsClient.
```

Only Colab and Kaggle have a store to ask. Everywhere else the environment is all there is. It writes into the environment because that is where the model and tracing SDKs go looking for a key. When it cannot find the secret it raises `RuntimeError` naming the fix for the host it detected, or returns `None` if you passed `required=False`. It never prints or logs the value.

Two things it deliberately does not do. It does not read `.env` itself, because `load_dotenv` already does that well and calling it first works fine. It does not prompt on stdin, because a library that blocks on input is painful in scripts and in CI.

### `Runtime`

Every side effect the package performs, in one injectable frozen dataclass. Substitute it in tests:

```python
from langsmith_studio_nb import Runtime, start_studio

start_studio(runtime=Runtime(status=lambda url: 200))
```

## Notes

- **No IPython dependency.** The package uses whatever IPython your notebook already has, so installing it will not upgrade Colab's pinned `ipython==7.34.0` out from under `google-colab`.
- **Studio must trust the tunnel domain.** On first connect, Studio blocks hosts it does not know. Add `*.trycloudflare.com` under Advanced Settings → Allowed Domains. Use the wildcard, not the exact host Studio offers to add for you, because that hostname changes every time a tunnel opens. The list lives in your browser's local storage, so it is per user and per browser, and there is no workspace-level setting. `start_studio` prints this reminder whenever it tunnels.
- **Restarting keeps the tunnel.** Cloudflare rate limits quick tunnels per IP address (`error code: 1015`, `429 Too Many Requests`), and every notebook on a host shares that address. A demo that restarts a few times can stop getting tunnels for a while. So a restart on the same port reuses the tunnel already running. The link does not change, and the restart takes about a second instead of five. Before reusing one, `start_studio` checks that it still answers. Cloudflare drops a quick tunnel on its own, and `cloudflared` keeps running behind a hostname that has stopped resolving, so a dead tunnel gets replaced rather than handed back to you. One case is exempt. If the kernel could never reach the tunnel to begin with, silence from it says nothing new. `stop_studio()` gives the tunnel up, and the next `start_studio()` opens a fresh one.
- **A failed tunnel is an error, not a link.** When `cloudflared` reports no URL, `start_studio` stops the local server and raises, rather than printing a link that can only fail to fetch.
- **`start_studio` replaces a tunnel that comes up dead.** Some quick tunnels never route. It asks each new one for the server before handing it to Studio, and opens another if Cloudflare answers that the tunnel routes nowhere. It will do that three times at most, because every attempt spends a tunnel against the rate limit. Silence means something different. A new hostname often takes minutes to resolve from the kernel while the browser reaches it straight away, so an unanswered tunnel gets used as it is rather than thrown away on a guess. If all three answer that they route nowhere, `start_studio` raises instead of printing a link that cannot work.
- **A tunnel URL is public and unauthenticated** for as long as the cell runs. Fine for a demo agent. Think twice with anything sensitive. The tunnel dies with the kernel.
- **Edits need a restart.** The server registers the graph object when it boots, so re-run `start_studio()` after changing your agent. Hot reload is not available from a notebook.
- **Colab drops idle runtimes** after roughly 90 minutes. You get a new tunnel URL after reconnecting.
- **Kaggle needs internet enabled** in the notebook settings, otherwise the tunnel cannot start.

## Development

```bash
uv sync
uv run ruff format . && uv run ruff check . && uv run ty check
uv run pytest                    # unit tests, 100% coverage enforced
uv run pytest -m integration     # boots a real agent server
```
