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
