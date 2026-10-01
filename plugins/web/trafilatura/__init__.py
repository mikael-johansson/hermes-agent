"""trafilatura local extract provider — bundled, auto-loaded.

Backed by the ``trafilatura`` Python package, which extracts the main text of a
page locally from raw HTML. No API key required and no third-party service —
it is a purely local, privacy-preserving extractor (the only thing that leaves
the box is the plain HTTP request to the target page).

``trafilatura`` is an optional dependency; :meth:`is_available` reflects whether
the package is importable. The plugin still registers either way so ``hermes
tools`` can prompt the user to install it.
"""

from __future__ import annotations

from plugins.web.trafilatura.provider import TrafilaturaProvider


def register(ctx) -> None:
    """Register the trafilatura provider with the plugin context."""
    ctx.register_web_search_provider(TrafilaturaProvider())
