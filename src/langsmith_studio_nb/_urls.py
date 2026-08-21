"""Build the LangGraph Studio URL for a running agent server."""

from __future__ import annotations

from urllib.parse import urlsplit

STUDIO_ORIGIN = "https://smith.langchain.com"


def port_of(url: str) -> int | None:
    """Return the port `url` names, or None when it names none."""
    return urlsplit(url).port


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
