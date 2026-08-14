"""Render the Studio link as plain text, clickable where the host allows it."""

from __future__ import annotations

from html import escape

LABEL = "Open LangGraph Studio:"


def link_html(url: str, *, hint: str | None = None) -> str:
    """Return the Studio link as an anchor, optionally followed by a hint."""
    safe = escape(url, quote=True)
    html = f'{LABEL} <a href="{safe}" target="_blank" rel="noopener">{safe}</a>'
    if hint:
        html = f"{html}<br>{escape(hint)}"
    return html


def link_text(url: str, *, hint: str | None = None) -> str:
    """Return the same link and hint as plain text, for hosts without HTML output."""
    text = f"{LABEL} {url}"
    return f"{text}\n{hint}" if hint else text
