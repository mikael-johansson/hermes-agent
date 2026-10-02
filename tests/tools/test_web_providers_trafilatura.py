"""Tests for the trafilatura local extract provider.

Covers:
- Capability flags: name / is_available (keyless, import-gated) /
  supports_search=False / supports_extract=True.
- extract() happy path against a local fixture HTML file: runs the REAL
  trafilatura pipeline (network hop monkeypatched to return the fixture bytes)
  and asserts clean markdown + the contract result shape.
- Per-URL SSRF gate (private/internal URL → blocked item, no fetch).
- Per-URL website-policy gate (blocked by rule → blocked item).
- Missing-package path: extract() returns an install-hint item, not a crash.

The network fetch is monkeypatched so the test is fully hermetic (no DNS, no
HTTP, no third-party API). Only the trafilatura extraction itself runs for
real, which is exactly what the task asks us to validate against a fixture.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from plugins.web.trafilatura import provider as traf_mod
from plugins.web.trafilatura.provider import TrafilaturaProvider

FIXTURE = Path(__file__).parent / "fixtures" / "trafilatura_article.html"
FIXTURE_HTML = FIXTURE.read_bytes()
FIXTURE_URL = "https://fixture.local/article"


def _run(coro) -> Any:
    """Run an async provider method on a fresh event loop."""
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Capability / availability
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_name(self):
        assert TrafilaturaProvider().name == "trafilatura"

    def test_implements_web_search_provider(self):
        from agent.web_search_provider import WebSearchProvider

        assert issubclass(TrafilaturaProvider, WebSearchProvider)

    def test_supports_extract_not_search(self):
        p = TrafilaturaProvider()
        assert p.supports_extract() is True
        assert p.supports_search() is False

    def test_is_available_true_when_package_installed(self):
        # trafilatura is installed in the test venv; the import-only probe must pass.
        assert TrafilaturaProvider().is_available() is True

    def test_is_available_false_when_import_fails(self):
        with patch.object(traf_mod, "_load_trafilatura", side_effect=ImportError("no trafilatura")):
            assert TrafilaturaProvider().is_available() is False


# ---------------------------------------------------------------------------
# extract() happy path against the local fixture (real trafilatura, fake network)
# ---------------------------------------------------------------------------


class TestExtractFixture:
    def _monkeypatch(self, monkeypatch):
        # Hermetic: no DNS / HTTP. Pretend the URL is public and allowed by policy.
        monkeypatch.setattr(traf_mod, "is_safe_url", lambda _url: True)
        monkeypatch.setattr(traf_mod, "check_website_access", lambda _url: None)
        monkeypatch.setattr(traf_mod, "_fetch_url", lambda _url: FIXTURE_HTML)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

    def test_returns_contract_shape(self, monkeypatch):
        self._monkeypatch(monkeypatch)
        results = _run(TrafilaturaProvider().extract([FIXTURE_URL]))
        assert isinstance(results, list)
        assert len(results) == 1
        item = results[0]
        # Contract keys (web_search_provider.py:153-184).
        for key in ("url", "title", "content", "raw_content", "metadata"):
            assert key in item, f"missing contract key: {key}"
        assert "error" not in item
        assert item["url"] == FIXTURE_URL
        assert isinstance(item["metadata"], dict)
        assert isinstance(item["content"], str)
        assert item["content"] == item["raw_content"]

    def test_extracted_clean_markdown(self, monkeypatch):
        self._monkeypatch(monkeypatch)
        content = _run(TrafilaturaProvider().extract([FIXTURE_URL]))[0]["content"]
        assert content.strip(), "extracted content should be non-empty"
        # Real trafilatura output: main-article text present...
        assert "Web scraping is the automated extraction of data" in content
        # ...and navigation / footer boilerplate stripped out.
        assert "Terms of Service" not in content
        assert "Privacy Policy" not in content
        assert "RSS Feed" not in content

    def test_metadata_title(self, monkeypatch):
        self._monkeypatch(monkeypatch)
        item = _run(TrafilaturaProvider().extract([FIXTURE_URL]))[0]
        # trafilatura uses the main article heading (<h1>) as the title, not the
        # <title> tag — assert the value it actually surfaces.
        assert item["title"] == "The History of Web Scraping"

    def test_multiple_urls_return_multiple_items(self, monkeypatch):
        self._monkeypatch(monkeypatch)
        results = _run(TrafilaturaProvider().extract([FIXTURE_URL, FIXTURE_URL]))
        assert len(results) == 2
        assert all("error" not in r for r in results)


# ---------------------------------------------------------------------------
# Per-URL SSRF + website-policy gates (no fetch should happen)
# ---------------------------------------------------------------------------


class TestSecurityGates:
    def test_ssrf_blocked_private_url(self, monkeypatch):
        # is_safe_url=False (private/internal) → blocked item, and _fetch_url never called.
        monkeypatch.setattr(traf_mod, "is_safe_url", lambda _url: False)
        fetch_called = {"n": 0}
        monkeypatch.setattr(traf_mod, "_fetch_url", lambda _url: fetch_called.__setitem__("n", fetch_called["n"] + 1))
        results = _run(TrafilaturaProvider().extract(["http://169.254.169.254/latest/meta-data/"]))
        assert len(results) == 1
        assert "error" in results[0]
        assert "private" in results[0]["error"].lower()
        assert fetch_called["n"] == 0  # network never touched

    def test_website_policy_blocked(self, monkeypatch):
        monkeypatch.setattr(traf_mod, "is_safe_url", lambda _url: True)
        monkeypatch.setattr(
            traf_mod,
            "check_website_access",
            lambda _url: {"host": "bad.example", "rule": "exact", "source": "config", "message": "Blocked by user policy"},
        )
        monkeypatch.setattr(traf_mod, "_fetch_url", lambda _url: FIXTURE_HTML)
        results = _run(TrafilaturaProvider().extract(["https://bad.example/page"]))
        assert len(results) == 1
        assert "error" in results[0]
        assert "policy" in results[0]["error"].lower()
        assert results[0]["blocked_by_policy"]["host"] == "bad.example"


# ---------------------------------------------------------------------------
# Missing-package path (is_available False) → install hint, not a crash
# ---------------------------------------------------------------------------


class TestMissingPackage:
    def test_extract_returns_install_hint(self, monkeypatch):
        with patch.object(traf_mod, "_load_trafilatura", side_effect=ImportError("no trafilatura")):
            # Force is_available() False inside extract() by patching the module probe.
            monkeypatch.setattr(traf_mod, "_trafilatura_importable", lambda: False)
            results = _run(TrafilaturaProvider().extract([FIXTURE_URL]))
        assert len(results) == 1
        assert "trafilatura package is not installed" in results[0]["error"]


# ---------------------------------------------------------------------------
# PDF branch: magic-byte routing + local PyMuPDF text extraction
# ---------------------------------------------------------------------------


def _make_text_pdf(title: str = "", body: str = "Hello PDF world.") -> bytes:
    """Generate an in-memory, single-page PDF with an embedded text layer."""
    import pymupdf

    doc = pymupdf.open()
    if title:
        doc.set_metadata({"title": title})
    page = doc.new_page()
    page.insert_text((72, 72), body)
    data = doc.tobytes()
    doc.close()
    return data


def _make_no_text_pdf() -> bytes:
    """Generate a minimal valid PDF with no text (scanned/image-only analogue)."""
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()  # blank page: no text, no drawings
    data = doc.tobytes()
    doc.close()
    return data


PDF_URL = "https://example.com/papers/2101.00075v1.pdf"


class TestPdfBranch:
    def test_is_pdf_magic_bytes(self):
        assert traf_mod._is_pdf(b"%PDF-1.5 rest") is True
        assert traf_mod._is_pdf(bytearray(b"%PDF-1.7")) is True
        # Negative: HTML and other non-PDF buffers are not treated as PDF.
        assert traf_mod._is_pdf(b"<html><body>hi</body></html>") is False
        assert traf_mod._is_pdf(b"\x00\x01\x02") is False
        assert traf_mod._is_pdf(b"%PDFX1.5") is False  # wrong 4th char

    def test_fixture_pdf_extract_helper_text_and_page_count(self):
        data = _make_text_pdf(title="A Fixture Paper", body="Hello PDF world.")
        text, metadata = traf_mod._extract_pdf_text(data, PDF_URL)
        assert "Hello PDF world." in text
        assert text.strip(), "extracted PDF text must be non-empty"
        assert metadata["pdf"] is True
        assert metadata["page_count"] == 1
        # Title comes from the PDF's own metadata.
        assert metadata["title"] == "A Fixture Paper"

    def test_title_falls_back_to_url_basename(self):
        data = _make_text_pdf(title="", body="No title here.")
        _, metadata = traf_mod._extract_pdf_text(data, PDF_URL)
        assert metadata["title"] == "2101.00075v1.pdf"

    def test_no_text_pdf_raises_scanned_error(self):
        data = _make_no_text_pdf()
        with pytest.raises(ValueError, match=r"no embedded text"):
            traf_mod._extract_pdf_text(data, PDF_URL)

    def test_pymupdf_missing_raises_lookup_error(self, monkeypatch):
        # Simulate a checkout without PyMuPDF: _load_pymupdf raises ImportError.
        monkeypatch.setattr(
            traf_mod, "_load_pymupdf",
            lambda: (_ for _ in ()).throw(ImportError("no pymupdf")),
        )
        data = _make_text_pdf()
        with pytest.raises(LookupError, match="pymupdf not installed"):
            traf_mod._extract_pdf_text(data, PDF_URL)

    def test_extract_pdf_end_to_end_contract(self, monkeypatch):
        # Hermetic: fake network returns an in-test PDF; real PyMuPDF runs.
        monkeypatch.setattr(traf_mod, "is_safe_url", lambda _url: True)
        monkeypatch.setattr(traf_mod, "check_website_access", lambda _url: None)
        monkeypatch.setattr(traf_mod, "_fetch_url", lambda _url: _make_text_pdf(title="E2E Paper"))
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
        results = _run(TrafilaturaProvider().extract([PDF_URL]))
        assert len(results) == 1
        item = results[0]
        for key in ("url", "title", "content", "raw_content", "metadata"):
            assert key in item, f"missing contract key: {key}"
        assert "error" not in item
        assert "Hello PDF world." in item["content"]
        assert item["metadata"]["pdf"] is True
        assert item["metadata"]["page_count"] == 1

    def test_extract_scanned_pdf_error_item(self, monkeypatch):
        monkeypatch.setattr(traf_mod, "is_safe_url", lambda _url: True)
        monkeypatch.setattr(traf_mod, "check_website_access", lambda _url: None)
        monkeypatch.setattr(traf_mod, "_fetch_url", lambda _url: _make_no_text_pdf())
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
        results = _run(TrafilaturaProvider().extract([PDF_URL]))
        assert len(results) == 1
        item = results[0]
        assert "error" in item
        assert "no embedded text (scanned/image-only PDF; OCR not supported by web_extract)" in item["error"]

    def test_html_not_routed_to_pdf_branch(self, monkeypatch):
        # HTML fixture must keep flowing to trafilatura, never the PDF helper.
        pdf_calls = {"n": 0}
        real_pdf = traf_mod._extract_pdf_text

        def spy_pdf(data, url):
            pdf_calls["n"] += 1
            return real_pdf(data, url)

        self._monkeypatch_html(monkeypatch, spy_pdf)
        results = _run(TrafilaturaProvider().extract([FIXTURE_URL]))
        assert len(results) == 1
        assert "error" not in results[0]
        assert "Web scraping is the automated extraction of data" in results[0]["content"]
        assert pdf_calls["n"] == 0  # PDF path never taken for HTML

    @staticmethod
    def _monkeypatch_html(monkeypatch, pdf_fn):
        monkeypatch.setattr(traf_mod, "is_safe_url", lambda _url: True)
        monkeypatch.setattr(traf_mod, "check_website_access", lambda _url: None)
        monkeypatch.setattr(traf_mod, "_fetch_url", lambda _url: FIXTURE_HTML)
        monkeypatch.setattr(traf_mod, "_extract_pdf_text", pdf_fn)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
