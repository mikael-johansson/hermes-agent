"""trafilatura local extract provider — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`.

This is a **local, keyless, privacy-preserving** content-extraction backend.
Unlike firecrawl/tavily (which ship the page to a third-party API), trafilatura
extracts the main text of a page *locally* from the raw HTML we fetch ourselves
with httpx. No API key, no third-party service, nothing leaves the box except
the plain HTTP request to the target page.

Availability mirrors the ``ddgs`` precedent: :meth:`is_available` is simply
"the ``trafilatura`` package is importable" (it is an optional dep, gated so a
fresh checkout without the package still boots and ``hermes tools`` can prompt
the user to install it).

Capabilities: :meth:`supports_extract` is True; :meth:`supports_search` is
False — search stays on ddgs/searxng.

Per-URL pipeline (see :meth:`extract`):
  1. SSRF gate — :func:`tools.url_safety.is_safe_url` (private/internal block)
     + :func:`tools.website_policy.check_website_access` (user blocklist), the
     same two gates the firecrawl provider applies (provider.py:464,486).
  2. Fetch — httpx GET with a browser User-Agent and follow_redirects, run in
     a worker thread and bounded by a per-URL ``asyncio.wait_for`` timeout
     (~60s) so a hung page cannot wedge the agent loop.
  3. Extract — branch on the fetched bytes:
       - PDF (magic bytes ``b'%PDF-'``) → local text extraction with PyMuPDF
         (all pages joined; page count + ``pdf: true`` in metadata; title
         from the PDF's own metadata or the URL basename);
       - HTML → ``trafilatura.extract(output_format='markdown',
         with_metadata=True)`` → (markdown, metadata).
     Both libraries are lazy-imported so there is no cold-start cost on a
     checkout without them.

The heavy helpers are module-level functions (``_fetch_url`` /
``_extract_markdown``) rather than closures so unit tests can monkeypatch the
network hop and exercise the *real* trafilatura pipeline against a local
fixture HTML file.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from agent.web_search_provider import WebSearchProvider
from tools.url_safety import is_safe_url
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)


# Per-URL wall-clock cap for the fetch step. trafilatura's local parse is fast
# (tens of ms for typical pages) so the budget is dominated by the network
# round-trip; 60s matches the firecrawl provider's per-URL scrape budget.
_PER_URL_TIMEOUT_SECS = 60

# A realistic desktop Chrome User-Agent so well-behaved static sites serve the
# same payload a browser would. (Hard anti-bot sites — e.g. BBC returning 403
# to scraper UAs — are outside what any local extractor can do; that is a
# site-side restriction, not a trafilatura defect.)
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

_ACCEPT_HEADER = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"

# trafilatura is an optional dep and its import tree is non-trivial, so we only
# import it on first real use (and only then when it is available). The result
# is cached on a module global so we pay the import once per process.
_TRAFILATURA_MOD: Optional[Any] = None

# PDF magic bytes: every well-formed PDF file starts with b'%PDF-'.
_PDF_MAGIC = b"%PDF-"


def _is_pdf(data: bytes) -> bool:
    """True when the fetched buffer starts with the PDF magic bytes."""
    return isinstance(data, (bytes, bytearray)) and bytes(data[:5]) == _PDF_MAGIC


def _load_trafilatura() -> Any:
    """Import and cache the ``trafilatura`` module.

    Raises ``ImportError`` when the package is not installed so the caller can
    surface a clear install hint.
    """
    global _TRAFILATURA_MOD
    if _TRAFILATURA_MOD is None:
        import trafilatura  # noqa: WPS433 — deliberately lazy

        _TRAFILATURA_MOD = trafilatura
    return _TRAFILATURA_MOD


def _trafilatura_importable() -> bool:
    """Cheap, import-only availability probe (no network)."""
    try:
        _load_trafilatura()
        return True
    except Exception:  # noqa: BLE001 — any import failure = unavailable
        return False


def _fetch_url(url: str) -> bytes:
    """Fetch the raw HTML for *url* over HTTP(S).

    Module-level (not a closure) so unit tests can monkeypatch it to return a
    local fixture's bytes and exercise the real extraction pipeline without a
    network. Uses httpx with a browser UA and redirect-following.
    """
    import httpx

    headers = {
        "User-Agent": _BROWSER_UA,
        "Accept": _ACCEPT_HEADER,
        "Accept-Language": "en-US,en;q=0.9",
    }
    with httpx.Client(
        timeout=_PER_URL_TIMEOUT_SECS,
        follow_redirects=True,
        headers=headers,
    ) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.content


def _split_frontmatter(text: str) -> tuple[str, Dict[str, Any]]:
    """Split a leading YAML front-matter block off markdown *text*.

    trafilatura's markdown output, when requested with ``with_metadata=True``
    and a specific ``output_format``, is a single *string* with the metadata
    serialized as a leading ``---\\nkey: value\\n---\\n`` block (NOT a tuple,
    which is what the default/undecorated output format returns). Recover both
    the metadata dict and the body. Returns the text unchanged with empty
    metadata when no front-matter block is present.
    """
    if not text or not text.lstrip().startswith("---"):
        return text, {}
    lines = text.split("\n")
    if lines[0].strip() != "---":
        return text, {}
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return text, {}
    meta: Dict[str, Any] = {}
    for line in lines[1:end]:
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if k:
            meta[k] = v
    body = "\n".join(lines[end + 1:]).lstrip("\n")
    return body, meta


def _extract_markdown(html: bytes) -> tuple[str, Dict[str, Any]]:
    """Run trafilatura over raw HTML; return (markdown, metadata dict).

    ``with_metadata=True`` changes the return shape by output format:
      - default / undecorated output → a ``(content, metadata)`` tuple;
      - a specific ``output_format`` (we use ``"markdown"``) → a single string
        with the metadata serialized as a leading YAML front-matter block.
    We handle both so the provider is robust across trafilatura versions.
    ``include_comments=False`` / ``favor_precision=True`` trade a little recall
    for precision (cleaner main-text separation).
    """
    mod = _load_trafilatura()
    result = mod.extract(
        html,
        output_format="markdown",
        with_metadata=True,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    if result is None:
        # trafilatura returns None (not a tuple/string) when it finds no content.
        return "", {}
    if isinstance(result, tuple):
        content, metadata = result
        if not isinstance(metadata, dict):
            metadata = {}
        return (content or ""), metadata
    # String with (possibly) a leading front-matter metadata block.
    content, metadata = _split_frontmatter(str(result))
    return (content or ""), metadata


def _load_pymupdf() -> Any:
    """Import and cache PyMuPDF.

    Tries the modern ``pymupdf`` name first (PyMuPDF >= 1.24), falling back to
    the deprecated ``fitz`` alias for older installs. Lazy + cached so there is
    no cold-start cost on a checkout without the package. Raises
    ``ImportError`` when neither module is importable so the caller can
    surface a clear install hint.
    """
    global _PYMUPDF_MOD
    if _PYMUPDF_MOD is None:
        try:
            import pymupdf  # noqa: WPS433 — deliberately lazy

            _PYMUPDF_MOD = pymupdf
        except ImportError:
            import fitz  # deprecated alias; kept for pre-1.24 installs

            _PYMUPDF_MOD = fitz
    return _PYMUPDF_MOD


_PYMUPDF_MOD: Optional[Any] = None


def _extract_pdf_text(data: bytes, url: str) -> tuple[str, Dict[str, Any]]:
    """Extract text from raw PDF *data* with PyMuPDF; return (text, metadata).

    Joins the text of every page (page text extracted with
    ``page.get_text()``). Metadata carries the page count, a ``pdf: True``
    flag, and the title from the PDF's own ``/Title`` field when present
    (otherwise the URL's basename).

    Raises ``LookupError`` (message 'pymupdf not installed') when PyMuPDF is
    not importable, and ``ValueError`` (scanned-PDF message) when the PDF has
    no embedded text layer — callers map these to the corresponding per-URL
    error items.
    """
    try:
        mod = _load_pymupdf()
    except ImportError:
        raise LookupError("pymupdf not installed") from None

    doc = mod.open(stream=bytes(data))
    try:
        pages = [page.get_text() for page in doc]
        text = "\n".join(pages).strip()
        if not text:
            # No embedded text layer: scanned / image-only PDF. web_extract
            # does not OCR.
            raise ValueError(
                "no embedded text (scanned/image-only PDF; OCR not supported by web_extract)"
            )
        pdf_meta = doc.metadata or {}
        title = (pdf_meta.get("title") or "").strip()
        if not title:
            base = url.rsplit("/", 1)[-1]
            title = base.rsplit("?", 1)[0]
        metadata: Dict[str, Any] = dict(pdf_meta)
        metadata["title"] = title
        metadata["pdf"] = True
        metadata["page_count"] = int(doc.page_count)
        return text, metadata
    finally:
        doc.close()


class TrafilaturaProvider(WebSearchProvider):
    """Local, keyless web-content extraction via the ``trafilatura`` package."""

    @property
    def name(self) -> str:
        return "trafilatura"

    @property
    def display_name(self) -> str:
        return "trafilatura (local, no key)"

    def is_available(self) -> bool:
        """True when the ``trafilatura`` package is importable (keyless).

        Cheap, import-only probe — must NOT do network I/O. Runs at
        tool-registration time and on every ``hermes tools`` paint.
        """
        return _trafilatura_importable()

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract clean markdown from one or more URLs locally.

        Per-URL pipeline: SSRF gate → website-policy gate → httpx fetch (worker
        thread, ~60s cap) → trafilatura markdown extraction. Per-URL failures
        (block, timeout, fetch/parse error) become items with an ``error`` field
        rather than raising, matching the firecrawl provider's contract.

        Returns the legacy per-URL list-of-results shape::

            [{"url", "title", "content", "raw_content", "metadata", ["error"]}, ...]
        """
        from tools.interrupt import is_interrupted as _is_interrupted

        if not self.is_available():
            return [
                {
                    "url": u,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "metadata": {},
                    "error": "trafilatura package is not installed — run `pip install trafilatura`",
                }
                for u in urls
            ]

        results: List[Dict[str, Any]] = []

        for url in urls:
            if _is_interrupted():
                results.append({"url": url, "title": "", "content": "", "raw_content": "", "metadata": {}, "error": "Interrupted"})
                continue

            # Pre-fetch SSRF gate: block private/internal network addresses.
            if not is_safe_url(url):
                logger.info("Blocked web_extract (SSRF) for %s", url)
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "metadata": {},
                        "error": "Blocked: URL targets a private or internal network address",
                    }
                )
                continue

            # Pre-fetch user website-policy gate.
            blocked = check_website_access(url)
            if blocked:
                logger.info(
                    "Blocked web_extract for %s by rule %s",
                    blocked["host"],
                    blocked["rule"],
                )
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "metadata": {},
                        "error": blocked["message"],
                        "blocked_by_policy": {
                            "host": blocked["host"],
                            "rule": blocked["rule"],
                            "source": blocked["source"],
                        },
                    }
                )
                continue

            # Fetch in a worker thread, bounded by a hard per-URL deadline.
            try:
                html = await asyncio.wait_for(
                    asyncio.to_thread(_fetch_url, url),
                    timeout=_PER_URL_TIMEOUT_SECS,
                )
            except asyncio.TimeoutError:
                logger.warning("trafilatura fetch timed out for %s", url)
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "metadata": {},
                        "error": f"Fetch timed out after {_PER_URL_TIMEOUT_SECS}s — page may be too large or unresponsive.",
                    }
                )
                continue
            except Exception as fetch_err:  # noqa: BLE001
                logger.debug("trafilatura fetch failed for %s: %s", url, fetch_err)
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "metadata": {},
                        "error": f"Fetch failed: {fetch_err}",
                    }
                )
                continue

            # Local extraction, branched on content type.
            if _is_pdf(html):
                # PDF → local PyMuPDF text extraction (not trafilatura).
                try:
                    content, metadata = _extract_pdf_text(html, url)
                except LookupError as pdf_mod_err:
                    results.append(
                        {
                            "url": url,
                            "title": "",
                            "content": "",
                            "raw_content": "",
                            "metadata": {"pdf": True},
                            "error": str(pdf_mod_err),
                        }
                    )
                    continue
                except ValueError as pdf_text_err:
                    results.append(
                        {
                            "url": url,
                            "title": "",
                            "content": "",
                            "raw_content": "",
                            "metadata": {"pdf": True},
                            "error": str(pdf_text_err),
                        }
                    )
                    continue
                except Exception as pdf_err:  # noqa: BLE001
                    logger.debug("PDF extraction failed for %s: %s", url, pdf_err)
                    results.append(
                        {
                            "url": url,
                            "title": "",
                            "content": "",
                            "raw_content": "",
                            "metadata": {},
                            "error": f"PDF extraction failed: {pdf_err}",
                        }
                    )
                    continue

                results.append(
                    {
                        "url": url,
                        "title": metadata.get("title", ""),
                        "content": content,
                        "raw_content": content,
                        "metadata": metadata,
                    }
                )
                continue

            # HTML → trafilatura.
            try:
                content, metadata = _extract_markdown(html)
            except Exception as extract_err:  # noqa: BLE001
                logger.debug("trafilatura extract failed for %s: %s", url, extract_err)
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "metadata": {},
                        "error": f"Extraction failed: {extract_err}",
                    }
                )
                continue

            title = metadata.get("title", "") or ""

            # Re-check SSRF safety after any redirect the fetch followed
            # (trafilatura may report the final URL in metadata['url']).
            final_url = (metadata.get("url") or url).strip()
            if final_url and final_url != url and not is_safe_url(final_url):
                logger.info("Blocked redirected web_extract for unsafe final URL: %s", final_url)
                results.append(
                    {
                        "url": final_url,
                        "title": title,
                        "content": "",
                        "raw_content": "",
                        "metadata": metadata,
                        "error": "Blocked: URL targets a private or internal network address",
                    }
                )
                continue

            if not content:
                results.append(
                    {
                        "url": final_url or url,
                        "title": title,
                        "content": "",
                        "raw_content": "",
                        "metadata": metadata,
                        "error": "No extractable content found (page may be JS-only or empty)",
                    }
                )
                continue

            results.append(
                {
                    "url": final_url or url,
                    "title": title,
                    "content": content,
                    "raw_content": content,
                    "metadata": metadata,
                }
            )

        return results

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "trafilatura (local, no key)",
            "badge": "free · no key · extract only",
            "tag": (
                "Local, privacy-preserving content extraction via the trafilatura "
                "Python package — no API key, nothing sent to a third-party API. "
                "Install with `pip install trafilatura`."
            ),
            "env_vars": [],
        }
