"""Render the Studio link as a button, with a plain-text fallback."""

from __future__ import annotations

_BUTTON = (
    '<a href="{url}" target="_blank" rel="noopener" '
    'style="display:inline-block;padding:12px 20px;background:#1C3C3C;color:#fff;'
    'border-radius:8px;font:600 15px sans-serif;text-decoration:none">'
    "&#127912; Open LangGraph Studio</a>"
)
_HINT = '<div style="margin-top:8px;font:400 13px sans-serif;color:#6b7280">{hint}</div>'


def link_html(url: str, *, hint: str | None = None) -> str:
    """Return the clickable Studio button, optionally followed by a hint."""
    html = _BUTTON.format(url=url)
    if hint:
        html = f"{html}{_HINT.format(hint=hint)}"
    return html


def link_text(url: str, *, hint: str | None = None) -> str:
    """Return the same link and hint as plain text, for hosts without HTML output."""
    return f"{url}\n{hint}" if hint else url
