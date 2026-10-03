"""Unit tests for the browser tab-reaping policy (T6).

Regression context (investigations/web-browser-tools-fix): the deployed
inactivity reaper fires at the 120 s timeout but closed 0 tabs because the
adopted-CDP branch protected the session's own last_url unconditionally
(verified live 2026-10-02: "CDP cleanup for task 2de6f117577c ... 41
targets, closed 0 stale page(s)").  This adds two reaping policies to
``tools/browser_tool.py``:

  * ``browser.max_tabs`` (default 20) — close the OLDEST page tabs (by
    tracked age) once the page count exceeds the cap.
  * ``browser.tab_ttl_seconds`` (default 86400) — close a page once its
    tracked age exceeds the TTL.

Pages have no CDP creation timestamp, so first-sighting ages are persisted
to ``~/.hermes/browser-page-ages.json`` (override with
``HERMES_BROWSER_AGE_FILE``).  Both the startup sweep and the inactivity
reaper's adopted-CDP branch drive closes through the single shared
chokepoint ``_plan_tab_reaping``.

Everything runs hermetically against a scratch in-process mock CDP endpoint
(no live browser, no host network).
"""

import json
import os
import threading
import time
import unittest.mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import tools.browser_tool as browser_tool


# ============================================================================
# Mock CDP endpoint (same in-process design as test_browser_cdp_stale_close)
# ============================================================================

def _target(tid: str, url: str, ttype: str = "page") -> dict:
    return {"id": tid, "type": ttype, "url": url, "title": ""}


class MockCdpServer:
    """In-process mock of a CDP endpoint: HTTP /json, /json/version and a
    browser-level websocket speaking Target.getTargets / Target.closeTarget.
    The target list persists across websocket connections, exactly like the
    real browser process the adopted endpoint points at."""

    def __init__(self, targets):
        self.targets = list(targets)
        self.connections = 0
        self.closed_ids = []
        self.http_thread = None
        self.loop_thread = None
        self.loop = None
        self._ws = None
        self._ready = threading.Event()
        self.http_url = None
        self.ws_url = None
        self._http_server = None
        self._stopping = False
        self.loop_exceptions = []

    def start(self):
        import inspect  # noqa: F401  (used by _shutdown for awaitable handling)
        self.http_thread = threading.Thread(target=self._http_serve, daemon=True)
        self.http_thread.start()
        self.loop = __import__("asyncio").new_event_loop()
        self.loop_thread = threading.Thread(target=self._loop_run, daemon=True)
        self.loop_thread.start()
        assert self._ready.wait(10), "mock CDP server did not start"

    def stop(self):
        self._stopping = True
        if self.loop is not None and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self._schedule_shutdown)
        if self.loop_thread is not None:
            self.loop_thread.join(timeout=5)
        if self._http_server is not None:
            self._http_server.shutdown()

    def _schedule_shutdown(self):
        import asyncio
        asyncio.ensure_future(self._shutdown())

    async def _shutdown(self):
        import inspect
        if self._ws is not None:
            result = self._ws.close()
            if inspect.isawaitable(result):
                await result

    def _loop_run(self):
        import asyncio
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._serve_ws())
        except Exception as exc:
            self.loop_exceptions.append(exc)
        finally:
            self.loop.close()

    def _http_serve(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, payload):
                data = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/json/version":
                    self._send({
                        "webSocketDebuggerUrl": outer.ws_url,
                        "Browser": "MockBrowser/1.0",
                    })
                elif self.path == "/json":
                    self._send(outer.targets)
                else:
                    self.send_response(404)
                    self.end_headers()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._http_server = server
        self.http_url = f"http://127.0.0.1:{server.server_address[1]}"
        server.serve_forever()

    async def _serve_ws(self):
        import asyncio
        import socket

        import websockets

        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        ws_port = probe.getsockname()[1]
        probe.close()

        async def handler(conn):
            self.connections += 1
            try:
                async for raw in conn:
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    mid = msg.get("id")
                    method = msg.get("method")
                    if method == "Target.getTargets":
                        await conn.send(json.dumps({
                            "id": mid,
                            "result": {"targetInfos": list(self.targets)},
                        }))
                    elif method == "Target.closeTarget":
                        tid = (msg.get("params") or {}).get("targetId")
                        removed = None
                        for t in list(self.targets):
                            if t["id"] == tid:
                                self.targets.remove(t)
                                removed = t
                                break
                        if removed is not None:
                            self.closed_ids.append(tid)
                        await conn.send(json.dumps({
                            "id": mid,
                            "result": {"success": removed is not None},
                        }))
                    else:
                        await conn.send(json.dumps({"id": mid, "error": {"message": "unknown method"}}))
            finally:
                pass

        self.ws_url = f"ws://127.0.0.1:{ws_port}/devtools/browser/mock"
        self._ws = await websockets.serve(handler, "127.0.0.1", ws_port)
        self._ready.set()

        closed = asyncio.ensure_future(self._ws.wait_closed())
        while True:
            if self._stopping:
                closed.cancel()
                self.loop.stop()
                return
            try:
                await asyncio.wait_for(asyncio.shield(closed), timeout=0.05)
                self.loop.stop()
                return
            except (asyncio.TimeoutError, asyncio.CancelledError):
                continue


@pytest.fixture()
def mock_cdp():
    """Scratch CDP endpoint with a mix of targets."""
    server = MockCdpServer([
        _target("old-1", "https://old.example/p1"),
        _target("old-2", "https://old.example/p2"),
        _target("new-1", "https://new.example/p1"),
        _target("blank-1", "about:blank"),
        _target("iframe-1", "https://old.example/p1", "iframe"),
        _target("worker-1", "blob:https://old.example", "worker"),
    ])
    server.start()
    yield server
    server.stop()


# ============================================================================
# Isolation: page-age cache + active-session globals
# ============================================================================

def _reset_age_cache():
    browser_tool._page_ages_cache = {}
    browser_tool._page_ages_cache_loaded = False


def _restore_globals(active, activity, sweep_flag, cleanup_running, thread):
    browser_tool._active_sessions.clear()
    browser_tool._active_sessions.update(active)
    browser_tool._session_last_activity.clear()
    browser_tool._session_last_activity.update(activity)
    browser_tool._cdp_startup_sweep_done = sweep_flag
    browser_tool._cleanup_running = cleanup_running
    browser_tool._cleanup_thread = thread
    _reset_age_cache()


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Point the age file at a scratch path and snapshot module globals."""
    age_file = str(tmp_path / "browser-page-ages.json")
    monkeypatch.setenv("HERMES_BROWSER_AGE_FILE", age_file)
    active = dict(browser_tool._active_sessions)
    activity = dict(browser_tool._session_last_activity)
    sweep_flag = browser_tool._cdp_startup_sweep_done
    cleanup_running = browser_tool._cleanup_running
    thread = browser_tool._cleanup_thread
    _reset_age_cache()
    yield age_file
    _restore_globals(active, activity, sweep_flag, cleanup_running, thread)


def _set_ages(age_file, mapping):
    """Directly write the age map to disk (and refresh the cache)."""
    Path(age_file).write_text(json.dumps(mapping))
    _reset_age_cache()


@pytest.fixture()
def age_file(tmp_path, monkeypatch):
    """A scratch age-file path (also set as HERMES_BROWSER_AGE_FILE)."""
    p = str(tmp_path / "browser-page-ages.json")
    monkeypatch.setenv("HERMES_BROWSER_AGE_FILE", p)
    _reset_age_cache()
    return p


# ============================================================================
# _plan_tab_reaping — the shared policy chokepoint
# ============================================================================

NOW = 1_000_000.0


class TestPlanTabReapingTtl:
    def test_over_ttl_closed(self):
        targets = [
            _target("a", "https://x/a"),
            _target("b", "https://x/b"),
        ]
        ages = {"https://x/a": NOW - 100_000, "https://x/b": NOW - 50_000}
        # TTL 86400: a (100000s) over, b (50000s) under.
        assert browser_tool._plan_tab_reaping(
            targets, max_tabs=0, ttl_seconds=86400, ages=ages, now=NOW,
        ) == ["a"]

    def test_under_ttl_kept(self):
        targets = [_target("b", "https://x/b")]
        ages = {"https://x/b": NOW - 100}
        assert browser_tool._plan_tab_reaping(
            targets, max_tabs=0, ttl_seconds=86400, ages=ages, now=NOW,
        ) == []

    def test_exact_boundary_not_over(self):
        targets = [_target("b", "https://x/b")]
        ages = {"https://x/b": NOW - 86400}
        # age == ttl is NOT "exceeds".
        assert browser_tool._plan_tab_reaping(
            targets, max_tabs=0, ttl_seconds=86400, ages=ages, now=NOW,
        ) == []

    def test_unknown_age_never_ttl_reaped(self):
        targets = [_target("b", "https://x/b")]
        # No entry in ages -> unknown -> never TTL-reaped.
        assert browser_tool._plan_tab_reaping(
            targets, max_tabs=0, ttl_seconds=86400, ages={}, now=NOW,
        ) == []


class TestPlanTabReapingMaxTabs:
    def _targets(self, n, base="https://list/"):
        return [_target(f"t{i}", f"{base}{i}") for i in range(n)]

    def test_zero_pages_no_trim(self):
        assert browser_tool._plan_tab_reaping(
            [], max_tabs=20, ttl_seconds=0, ages={}, now=NOW,
        ) == []

    def test_one_page_no_trim(self):
        targets = self._targets(1)
        assert browser_tool._plan_tab_reaping(
            targets, max_tabs=20, ttl_seconds=0, ages={}, now=NOW,
        ) == []

    def test_many_pages_trims_to_cap_oldest_first(self):
        targets = self._targets(5)  # t0..t4
        ages = {
            f"https://list/{i}": NOW - (1000 * (5 - i))
            for i in range(5)
        }
        # t0 oldest (5000s) ... t4 newest (1000s).
        # cap 3 -> close oldest t0, t1.
        plan = browser_tool._plan_tab_reaping(
            targets, max_tabs=3, ttl_seconds=0, ages=ages, now=NOW,
        )
        assert set(plan) == {"t0", "t1"}
        # Oldest-first ordering in the returned list.
        assert plan.index("t0") < plan.index("t1")

    def test_at_or_under_cap_no_trim(self):
        targets = self._targets(3)
        assert browser_tool._plan_tab_reaping(
            targets, max_tabs=3, ttl_seconds=0, ages={}, now=NOW,
        ) == []

    def test_protected_url_survives_max_tabs(self):
        targets = self._targets(4)
        ages = {f"https://list/{i}": NOW - (1000 * (4 - i)) for i in range(4)}
        # t0 oldest, but protected -> it survives; next-oldest t1, t2 closed.
        plan = browser_tool._plan_tab_reaping(
            targets, max_tabs=2, ttl_seconds=0, ages=ages,
            protect_urls=("https://list/0",), now=NOW,
        )
        assert "t0" not in plan
        assert set(plan) == {"t1", "t2"}

    def test_unknown_age_counts_as_newest_for_max_tabs(self):
        # t0 oldest, t1 unknown (newest), t2 in between.
        targets = [
            _target("t0", "https://u/0"),
            _target("t1", "https://u/1"),
            _target("t2", "https://u/2"),
        ]
        ages = {"https://u/0": NOW - 9000, "https://u/2": NOW - 3000}
        # cap 1: keep newest. Order oldest-first: t0(9000) > t2(3000) > t1(unknown).
        # Close oldest until 1 remains -> close t0 and t2, keep t1 (unknown=newest).
        plan = browser_tool._plan_tab_reaping(
            targets, max_tabs=1, ttl_seconds=0, ages=ages, now=NOW,
        )
        assert set(plan) == {"t0", "t2"}


class TestPlanTabReapingCombined:
    def test_ttl_then_max_tabs(self):
        # 4 pages: p0 very old (TTL-reaped), p1 old, p2/p3 fresh.
        targets = [
            _target("p0", "https://v/0"),
            _target("p1", "https://v/1"),
            _target("p2", "https://v/2"),
            _target("p3", "https://v/3"),
        ]
        ages = {
            "https://v/0": NOW - 200_000,  # over ttl
            "https://v/1": NOW - 10_000,
            "https://v/2": NOW - 5_000,
            "https://v/3": NOW - 1_000,
        }
        # TTL reaps p0. Then 3 remain; cap 2 -> close oldest remaining (p1).
        plan = browser_tool._plan_tab_reaping(
            targets, max_tabs=2, ttl_seconds=86400, ages=ages, now=NOW,
        )
        assert set(plan) == {"p0", "p1"}

    def test_both_disabled_closes_nothing(self):
        targets = [
            _target("a", "https://x/a"),
            _target("b", "https://x/b"),
        ]
        ages = {"https://x/a": NOW - 100_000, "https://x/b": NOW - 90_000}
        assert browser_tool._plan_tab_reaping(
            targets, max_tabs=0, ttl_seconds=0, ages=ages, now=NOW,
        ) == []

    def test_none_knobs_disabled(self):
        targets = [_target("a", "https://x/a")]
        ages = {"https://x/a": NOW - 100_000}
        assert browser_tool._plan_tab_reaping(
            targets, max_tabs=None, ttl_seconds=None, ages=ages, now=NOW,
        ) == []


class TestPlanTabReapingExclusions:
    def test_workers_iframes_blank_never_candidates(self):
        targets = [
            _target("w", "blob:https://x", "worker"),
            _target("f", "https://x/iframe", "iframe"),
            _target("b", "about:blank", "page"),
            _target("e", "", "page"),
            _target("ui", "devtools://x", "browser_ui"),
            # one real page, but at/under the cap and under ttl
            _target("ok", "https://x/ok", "page"),
        ]
        ages = {"https://x/ok": NOW - 10}
        assert browser_tool._plan_tab_reaping(
            targets, max_tabs=1, ttl_seconds=86400, ages=ages, now=NOW,
        ) == []

    def test_protected_never_closed_by_either_rule(self):
        targets = [
            _target("a", "https://x/a"),
            _target("prot", "https://keep/prot"),
        ]
        ages = {"https://x/a": NOW - 100_000, "https://keep/prot": NOW - 200_000}
        # prot is oldest AND over ttl, but protected -> never closed.
        plan = browser_tool._plan_tab_reaping(
            targets, max_tabs=1, ttl_seconds=86400, ages=ages,
            protect_urls=("https://keep/prot",), now=NOW,
        )
        assert plan == ["a"]

    def test_prefix_protection_both_directions(self):
        targets = [_target("a", "https://x/a/path")]
        ages = {"https://x/a/path": NOW - 100_000}
        # protect a prefix of the target URL.
        assert browser_tool._plan_tab_reaping(
            targets, max_tabs=0, ttl_seconds=10, ages=ages,
            protect_urls=("https://x/a",), now=NOW,
        ) == []


# ============================================================================
# Age-file persistence
# ============================================================================

class TestPageAgesFile:
    def test_corrupt_file_starts_fresh(self, age_file):
        Path(age_file).write_text("not json {{{")
        ages = browser_tool._update_page_ages(["https://x/1"], now=NOW)
        assert ages == {"https://x/1": NOW}

    def test_missing_file_starts_fresh(self, age_file):
        assert not Path(age_file).exists()
        ages = browser_tool._update_page_ages(["https://x/1"], now=NOW)
        assert ages == {"https://x/1": NOW}
        assert Path(age_file).exists()

    def test_records_new_drops_absent(self, age_file):
        _set_ages(age_file, {"https://gone/1": NOW - 5000, "https://keep/1": NOW - 9000})
        ages = browser_tool._update_page_ages(
            ["https://keep/1", "https://new/1"], now=NOW,
        )
        # keep/1 keeps its original first-seen; new/1 recorded; gone/1 dropped.
        assert ages["https://keep/1"] == NOW - 9000
        assert ages["https://new/1"] == NOW
        assert "https://gone/1" not in ages

    def test_blank_and_about_blank_not_tracked(self, age_file):
        ages = browser_tool._update_page_ages(
            ["about:blank", "", "https://real/1"], now=NOW,
        )
        assert ages == {"https://real/1": NOW}

    def test_cap_entry_count(self, age_file, monkeypatch):
        monkeypatch.setattr(
            browser_tool, "_PAGE_AGES_MAX_ENTRIES", 3, raising=False,
        )
        present = [f"https://m/{i}" for i in range(6)]
        ages = browser_tool._update_page_ages(present, now=NOW)
        assert len(ages) == 3
        # Keeps the most recent (all same epoch here -> keeps any 3).
        on_disk = json.loads(Path(age_file).read_text())
        assert len(on_disk) == 3

    def test_atomic_write_no_tmp_leftovers(self, age_file):
        browser_tool._update_page_ages(["https://x/1"], now=NOW)
        leftovers = [
            p for p in Path(age_file).parent.iterdir()
            if p.name.startswith(".browser-page-ages-") and p.name.endswith(".tmp")
        ]
        assert leftovers == []


# ============================================================================
# _get_tab_reaping_policy — config reading
# ============================================================================

class TestGetTabReapingPolicy:
    def test_defaults_when_no_config(self, monkeypatch):
        import hermes_cli.config as hc
        monkeypatch.setattr(hc, "read_raw_config", lambda: {}, raising=False)
        # No keys -> fall back to module defaults.
        assert browser_tool._get_tab_reaping_policy() == (
            browser_tool.DEFAULT_BROWSER_MAX_TABS,
            browser_tool.DEFAULT_BROWSER_TAB_TTL_SECONDS,
        )
        assert browser_tool.DEFAULT_BROWSER_MAX_TABS == 20
        assert browser_tool.DEFAULT_BROWSER_TAB_TTL_SECONDS == 86400

    def test_reads_config_values(self, monkeypatch):
        import hermes_cli.config as hc
        monkeypatch.setattr(
            hc, "read_raw_config",
            lambda: {"browser": {"max_tabs": 5, "tab_ttl_seconds": 3600}},
            raising=False,
        )
        assert browser_tool._get_tab_reaping_policy() == (5, 3600)

    def test_invalid_values_fall_back(self, monkeypatch):
        import hermes_cli.config as hc
        monkeypatch.setattr(
            hc, "read_raw_config",
            lambda: {"browser": {"max_tabs": "abc", "tab_ttl_seconds": -5}},
            raising=False,
        )
        assert browser_tool._get_tab_reaping_policy() == (
            browser_tool.DEFAULT_BROWSER_MAX_TABS,
            browser_tool.DEFAULT_BROWSER_TAB_TTL_SECONDS,
        )

    def test_read_raw_config_failure_falls_back(self, monkeypatch):
        import hermes_cli.config as hc

        def boom():
            raise RuntimeError("no config")
        monkeypatch.setattr(hc, "read_raw_config", boom, raising=False)
        assert browser_tool._get_tab_reaping_policy() == (
            browser_tool.DEFAULT_BROWSER_MAX_TABS,
            browser_tool.DEFAULT_BROWSER_TAB_TTL_SECONDS,
        )


# ============================================================================
# End-to-end: startup sweep reap via the mock CDP endpoint
# ============================================================================

class TestSweepReaping:
    def _patch_policy(self, monkeypatch, max_tabs, ttl):
        monkeypatch.setattr(
            browser_tool, "_get_tab_reaping_policy", lambda: (max_tabs, ttl),
        )

    def test_sweep_closes_over_ttl_keeps_fresh_and_active(self, mock_cdp, monkeypatch, age_file):
        monkeypatch.setenv("BROWSER_CDP_URL", mock_cdp.http_url)
        self._patch_policy(monkeypatch, max_tabs=100, ttl=86400)
        # two old pages (tracked 2 days ago), one fresh page, one active session page
        _set_ages(age_file, {
            "https://old.example/p1": NOW - 172800,
            "https://old.example/p2": NOW - 172800,
        })
        browser_tool._active_sessions["task-sweep"] = {
            "session_name": "cdp_sweep",
            "bb_session_id": None,
            "cdp_url": mock_cdp.ws_url,
            "last_url": "https://new.example/p1",
        }
        browser_tool._session_last_activity["task-sweep"] = time.time()

        with unittest.mock.patch("tools.browser_tool.time.sleep"):
            browser_tool._sweep_stale_cdp_targets()

        remaining = {t["id"] for t in mock_cdp.targets}
        # old pages closed; fresh + active + blank + iframe + worker survive.
        assert "old-1" not in remaining and "old-2" not in remaining
        assert "new-1" in remaining
        assert "blank-1" in remaining and "iframe-1" in remaining and "worker-1" in remaining
        assert sorted(mock_cdp.closed_ids) == ["old-1", "old-2"]

    def test_sweep_max_tabs_trims_to_cap(self, mock_cdp, monkeypatch, age_file):
        monkeypatch.setenv("BROWSER_CDP_URL", mock_cdp.http_url)
        self._patch_policy(monkeypatch, max_tabs=1, ttl=0)
        # old-1 oldest, old-2 next, new-1 newest.
        _set_ages(age_file, {
            "https://old.example/p1": NOW - 3000,
            "https://old.example/p2": NOW - 2000,
            "https://new.example/p1": NOW - 1000,
        })
        browser_tool._active_sessions["task-sweep"] = {
            "session_name": "cdp_sweep",
            "bb_session_id": None,
            "cdp_url": mock_cdp.ws_url,
            "last_url": "https://new.example/p1",
        }
        browser_tool._session_last_activity["task-sweep"] = time.time()

        with unittest.mock.patch("tools.browser_tool.time.sleep"):
            browser_tool._sweep_stale_cdp_targets()

        # cap 1, active (new-1) protected -> close oldest two, keep new-1.
        assert sorted(mock_cdp.closed_ids) == ["old-1", "old-2"]
        urls = {t["url"] for t in mock_cdp.targets}
        assert "https://new.example/p1" in urls

    def test_sweep_never_nukes_recent_tabs_of_live_session(self, mock_cdp, monkeypatch, age_file):
        # Regression: before T6, two concurrent webui tabs + a restart meant
        # the second tab got nuked.  Now only TTL/max-tabs violations close.
        monkeypatch.setenv("BROWSER_CDP_URL", mock_cdp.http_url)
        self._patch_policy(monkeypatch, max_tabs=100, ttl=86400)
        _set_ages(age_file, {})  # nothing tracked -> no TTL violations
        # two fresh live-session tabs
        browser_tool._active_sessions["t1"] = {
            "session_name": "a", "bb_session_id": None,
            "cdp_url": mock_cdp.ws_url, "last_url": "https://old.example/p1",
        }
        browser_tool._active_sessions["t2"] = {
            "session_name": "b", "bb_session_id": None,
            "cdp_url": mock_cdp.ws_url, "last_url": "https://old.example/p2",
        }
        with unittest.mock.patch("tools.browser_tool.time.sleep"):
            browser_tool._sweep_stale_cdp_targets()
        # Nothing closed: no TTL violations, under the cap, all protected.
        assert mock_cdp.closed_ids == []

    def test_sweep_noop_without_cdp_override(self, monkeypatch):
        monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
        with unittest.mock.patch("tools.browser_tool._cdp_ws_url_for") as mock_ws:
            browser_tool._sweep_stale_cdp_targets()
        mock_ws.assert_not_called()

    def test_sweep_dead_endpoint_swallowed(self, monkeypatch):
        monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:1")
        with unittest.mock.patch("tools.browser_tool.time.sleep"):
            browser_tool._sweep_stale_cdp_targets()  # must not raise


# ============================================================================
# End-to-end: inactivity reaper's adopted-CDP branch
# ============================================================================

class TestReaperAdoptedCdp:
    def _seed(self, mock_cdp, age_file, with_activity=True, last_url="https://new.example/p1"):
        browser_tool._active_sessions["task-cdp"] = {
            "session_name": "cdp_mock",
            "bb_session_id": None,
            "cdp_url": mock_cdp.ws_url,
            "features": {"cdp_override": True},
            "last_url": last_url,
        }
        if with_activity:
            browser_tool._session_last_activity["task-cdp"] = time.time()
        _set_ages(age_file, {
            "https://old.example/p1": NOW - 172800,
            "https://old.example/p2": NOW - 172800,
        })

    def test_inactive_session_closes_own_last_url_when_over_ttl(
        self, mock_cdp, monkeypatch, age_file,
    ):
        # THE regression: the reaper must no longer protect its own last_url
        # for an INACTIVE session, so that page becomes a normal candidate.
        monkeypatch.setattr(
            browser_tool, "_get_tab_reaping_policy", lambda: (100, 86400),
        )
        self._seed(mock_cdp, age_file, with_activity=True,
                   last_url="https://old.example/p1")
        with unittest.mock.patch("tools.browser_tool._run_browser_command") as mock_run:
            browser_tool._cleanup_single_browser_session("task-cdp")
        mock_run.assert_not_called()
        # Its own last_url (old-1, over ttl) is now closed — not protected.
        assert "old-1" in mock_cdp.closed_ids
        assert "old-2" in mock_cdp.closed_ids
        assert "task-cdp" not in browser_tool._active_sessions

    def test_inactive_session_trims_to_max_tabs(self, mock_cdp, monkeypatch, age_file):
        monkeypatch.setattr(
            browser_tool, "_get_tab_reaping_policy", lambda: (1, 0),
        )
        self._seed(mock_cdp, age_file, with_activity=True,
                   last_url="https://new.example/p1")
        # ages: old-1 oldest, old-2 next, new-1 newest (== last_url).
        browser_tool._session_last_activity["task-cdp"] = time.time()
        with unittest.mock.patch("tools.browser_tool._run_browser_command") as mock_run:
            browser_tool._cleanup_single_browser_session("task-cdp")
        mock_run.assert_not_called()
        # cap 1, no protection for inactive: keep newest (new-1), close the rest.
        assert sorted(mock_cdp.closed_ids) == ["old-1", "old-2"]
        urls = {t["url"] for t in mock_cdp.targets}
        assert "https://new.example/p1" in urls

    def test_expiry_replacement_keeps_protecting_last_url(self, mock_cdp, monkeypatch, age_file):
        # Expiry-replacement path: the session is being *renewed*, so the
        # in-use page (last_url) must survive -> protect_last_url=True.
        monkeypatch.setattr(
            browser_tool, "_get_tab_reaping_policy", lambda: (100, 86400),
        )
        self._seed(mock_cdp, age_file, with_activity=False,
                   last_url="https://old.example/p1")
        with unittest.mock.patch("tools.browser_tool._run_browser_command") as mock_run:
            browser_tool._cleanup_single_browser_session("task-cdp", protect_last_url=True)
        mock_run.assert_not_called()
        # last_url (old-1) protected -> NOT closed; the other over-ttl page is.
        assert "old-1" not in mock_cdp.closed_ids
        assert "old-2" in mock_cdp.closed_ids

    def test_non_cdp_session_still_uses_agent_browser_close(self, monkeypatch):
        browser_tool._active_sessions["task-local"] = {
            "session_name": "local_mock",
            "bb_session_id": None,
        }
        browser_tool._session_last_activity["task-local"] = time.time()
        with unittest.mock.patch(
            "tools.browser_tool._run_browser_command",
            return_value={"success": True},
        ) as mock_run:
            browser_tool._cleanup_single_browser_session("task-local")
        mock_run.assert_called_once_with("task-local", "close", [], timeout=10)
        assert "task-local" not in browser_tool._active_sessions


# ============================================================================
# _close_stale_cdp_targets with an explicit close_ids plan
# ============================================================================

class TestCloseWithPlan:
    def test_closes_exactly_the_planned_ids(self, mock_cdp):
        summary = browser_tool._close_stale_cdp_targets(
            mock_cdp.ws_url, close_ids=("old-1",),
        )
        assert summary["closed"] == 1
        assert mock_cdp.closed_ids == ["old-1"]
        # others untouched
        assert {t["id"] for t in mock_cdp.targets} == {
            "old-2", "new-1", "blank-1", "iframe-1", "worker-1",
        }

    def test_empty_plan_closes_nothing(self, mock_cdp):
        summary = browser_tool._close_stale_cdp_targets(
            mock_cdp.ws_url, close_ids=(),
        )
        assert summary["closed"] == 0
        assert mock_cdp.closed_ids == []

    def test_plan_ids_defensively_intersected(self, mock_cdp):
        # A plan id that isn't in the live target list is ignored.
        summary = browser_tool._close_stale_cdp_targets(
            mock_cdp.ws_url, close_ids=("ghost", "old-2"),
        )
        assert mock_cdp.closed_ids == ["old-2"]
        assert summary["closed"] == 1

    def test_legacy_no_plan_closes_all_real_pages(self, mock_cdp):
        # No close_ids -> legacy behavior: every non-blank page closed.
        summary = browser_tool._close_stale_cdp_targets(mock_cdp.ws_url)
        assert summary["closed"] == 3
        assert sorted(mock_cdp.closed_ids) == ["new-1", "old-1", "old-2"]
