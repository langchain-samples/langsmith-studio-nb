"""Render the Studio link as a button, with a plain-text fallback."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

_BUTTON = (
    '<a href="{url}" target="_blank" rel="noopener" '
    'style="display:inline-block;padding:12px 20px;background:#1C3C3C;color:#fff;'
    'border-radius:8px;font:600 15px sans-serif;text-decoration:none">'
    "&#127912; Open LangGraph Studio</a>"
)


def link_html(url: str) -> str:
    """Return the clickable Studio button as an HTML fragment."""
    return _BUTTON.format(url=url)


def render(url: str, *, display_html: Callable[[str], None], echo: Callable[[str], None]) -> None:
    """Show the Studio link, always echoing the raw URL as a fallback.

    Some notebook hosts strip rendered HTML, so the plain URL is not redundant.
    """
    display_html(link_html(url))
    echo(url)
