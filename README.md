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

On Colab, add both keys under the 🔑 **Secrets** panel in the left sidebar with **Notebook access** on, and read them from there:

```python
import os

from google.colab import userdata

os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"] = userdata.get("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
```

There is no `userdata` in your own Jupyter, so export those three variables before you start it instead.

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

[**examples/deep_agent_in_studio.ipynb**](examples/deep_agent_in_studio.ipynb) is the quickstart above as a runnable notebook, plus the two things Colab adds: where to keep your API keys, and the one Studio setting a tunnel needs. It runs top to bottom there, and in your own Jupyter.

The keys come from Colab's 🔑 Secrets panel. Running your own Jupyter, they come from the environment, or from a `.env` file you make with `cd examples && cp .env.example .env`. The template sits next to the notebook, because the keys belong to the example rather than to the package. The notebook calls `load_dotenv()`, which searches the kernel's working directory and then its parents, so keep the `.env` in `examples/` and start Jupyter there or anywhere below the repository root.

## Several agents at once

Pass every variable you want to demo and pick between them in Studio's graph menu:

```python
start_studio("planner", "writer")
```

One server, one tunnel, one link. Studio switches between the graphs without a restart.

## Where it works

Colab, and your own Jupyter. They differ in one thing.

On Colab the kernel runs on Google's machine, so the browser cannot reach it and `start_studio` opens a Cloudflare quick tunnel to carry Studio's requests. The `cloudflared` binary downloads itself the first time. In Jupyter, JupyterLab, or VS Code on your own machine, the browser is already there, so it serves `http://127.0.0.1:2024` and skips the tunnel.

Override that when the guess is wrong:

```python
start_studio(tunnel=True)  # e.g. a browser that blocks https -> localhost
start_studio(tunnel=False)  # e.g. a remote kernel you already expose yourself
```

## API

### `start_studio(*variables, port=2024, tunnel=None, timeout=180, verbose=False)`

Serves the compiled graphs bound to `variables` in the notebook namespace, `agent` by default, and displays a Studio link. Each variable becomes a graph of the same name. Returns a `StudioSession` carrying `api_url`, `studio_url`, `tunnel`, and `graphs`.

Pass `verbose=True` to see the agent server's logs. Raises `NameError` if a variable is undefined, `RuntimeError` if the server dies during startup or the tunnel never comes up, and `TimeoutError` if the server never answers. If the port you asked for is busy, it picks a free one instead.

### `stop_studio()`

Stops the running server and its tunnel, puts the log levels back, and gives `LANGGRAPH_API_URL` back to whatever set it. Safe to call when nothing is running, and it does not raise for anything short of an interrupt: a step that fails does not stop the ones after it, and a `KeyboardInterrupt` still lands, but only once everything has been given up. A server draining a long request can outlive the wait, and everything else is released either way. Waits for a `start_studio` already in flight rather than tearing down the session it is about to hand back. Use it to force the next `start_studio` onto a fresh tunnel.

### `Runtime`

Every side effect the package performs, in one injectable frozen dataclass. Substitute it in tests:

```python
from langsmith_studio_nb import Runtime, start_studio

start_studio(runtime=Runtime(status=lambda url: 200))
```

## Notes

- **No IPython dependency.** The package uses whatever IPython your notebook already has, so installing it will not upgrade Colab's pinned `ipython==7.34.0` out from under `google-colab`.
- **Studio must trust the tunnel domain.** On first connect, Studio blocks hosts it does not know. Add `*.trycloudflare.com` under Advanced Settings → Allowed Domains. Use the wildcard, not the exact host Studio offers to add for you, because that hostname changes every time a tunnel opens. The list lives in your browser's local storage, so it is per user and per browser, and there is no workspace-level setting. `start_studio` prints this reminder whenever it tunnels.
- **Restarting keeps the tunnel.** Cloudflare rate limits quick tunnels per IP address (`error code: 1015`, `429 Too Many Requests`), and every notebook on a Colab host shares that address. A demo that restarts a few times can stop getting tunnels for a while. So a restart on the same port reuses the tunnel already running. The link does not change, and the restart takes about a second instead of five. Before reusing one, `start_studio` asks cloudflared whether it still holds a connection to Cloudflare. The process keeps running behind a URL that has stopped working when Cloudflare drops a quick tunnel, so a dead one gets replaced rather than handed back to you. `stop_studio()` gives the tunnel up, and the next `start_studio()` opens a fresh one.
- **A failed tunnel is an error, not a link.** When `cloudflared` reports no URL, `start_studio` stops the local server and raises, rather than printing a link that can only fail to fetch.
- **`start_studio` replaces a tunnel that never connects.** cloudflared prints a `trycloudflare.com` URL as soon as Cloudflare assigns one, before it holds a connection to carry the traffic — and on a runtime that blocks its egress it never gets one, retrying forever behind a URL that answers for nobody. `start_studio` asks cloudflared itself, through `/ready` on its metrics server, and opens another tunnel if this one never connects. The second attempt asks for `--protocol http2`, which reaches port 7844 over TCP instead of the UDP the default needs — a runtime may drop one and not the other. There is no third: every attempt spends a quick tunnel against the rate limit, and a repeat of either protocol buys no route the first try did not already fail on. So it raises instead, rather than printing a link that cannot work.
- **The last word on a tunnel is the check just before the link.** A tunnel kept across a restart is asked again once the new server answers, because Cloudflare can drop one in the seconds a server takes to come back up. A tunnel that went down in that window is replaced rather than handed to you.
- **A tunnel URL is public and unauthenticated** for as long as the cell runs. Fine for a demo agent. Think twice with anything sensitive. The tunnel dies with the kernel.
- **Edits need a restart.** The server registers the graph object when it boots, so re-run `start_studio()` after changing your agent. Hot reload is not available from a notebook.
- **Colab drops idle runtimes** after roughly 90 minutes. You get a new tunnel URL after reconnecting.

## Development

```bash
uv sync
uv run ruff format . && uv run ruff check . && uv run ty check
uv run pytest                    # unit tests, 100% coverage enforced
uv run pytest -m integration --no-cov   # boots a real agent server
```
