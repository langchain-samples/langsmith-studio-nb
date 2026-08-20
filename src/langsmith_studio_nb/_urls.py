"""Build the LangGraph Studio URL for a running agent server."""

from __future__ import annotations

from urllib.parse import urlsplit

STUDIO_ORIGIN = "https://smith.langchain.com"

# 0.0.0.0 is matched here, never bound.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "0.0.0.0", "::1", "localhost"})  # noqa: S104


def is_loopback_url(url: str) -> bool:
    """Report whether `url` names this machine, and so is private to the kernel."""
    return urlsplit(url).hostname in _LOOPBACK_HOSTS


def studio_url(
    api_url: str, *, workspace_id: str | None = None, origin: str = STUDIO_ORIGIN
) -> str:
    """Return the Studio URL that opens `api_url`.

    Args:
        api_url: Base URL of the running agent server.
        workspace_id: LangSmith workspace to preselect, skipping the picker.
        origin: Studio origin, for self-hosted LangSmith instances.
    """
    # Studio expects baseUrl verbatim, matching what `langgraph dev` emits.
    # Do not percent-encode it.
    url = f"{origin.rstrip('/')}/studio/?baseUrl={api_url.rstrip('/')}"
    if workspace_id:
        url = f"{url}&organizationId={workspace_id}"
    return url
