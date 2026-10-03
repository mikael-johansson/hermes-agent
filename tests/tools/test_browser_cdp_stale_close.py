"""Hermetic tests for raw-CDP stale-target cleanup of adopted cdp_url sessions.

Regression context (investigations/web-browser-tools-fix/b_diag.md): the
inactivity reaper's ``agent-browser --cdp <url> close`` against the
container-owned browser (hermes-browser-box, Chromium = PID 1) hangs to the
10 s command timeout and closes nothing — tabs accumulated 117 deep.  The
verified fix is raw CDP over the browser websocket: ``Target.getTargets`` +
``Target.closeTarget`` (the ``websockets`` library sends no Origin header;
websocket-client's ``suppress_origin`` kwarg is NOT valid for it).

Everything here runs against a scratch in-process mock CDP endpoint (HTTP
/json + /json/version + a browser-level websocket) — no live browser, no
host network.
"""

import asyncio
import inspect
import json
import threading
import unittest.mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import tools.browser_tool as browser_tool


# ============================================================================
# Mock CDP endpoint
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

    # -- lifecycle ----------------------------------------------------------
    def start(self):
        self.http_thread = threading.Thread(target=self._http_serve, daemon=True)
        self.http_thread.start()
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._loop_run, daemon=True)
        self.loop_thread.start()
        assert self._ready.wait(10), "mock CDP server did not start"

    def stop(self):
        # Graceful teardown: signal the serve loop, schedule the websocket
        # server close on the loop, join the loop thread, then stop HTTP.
        self._stopping = True
        if self.loop is not None and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self._schedule_shutdown)
        if self.loop_thread is not None:
            self.loop_thread.join(timeout=5)
        if self._http_server is not None:
            self._http_server.shutdown()

    def _schedule_shutdown(self):
        asyncio.ensure_future(self._shutdown())

    async def _shutdown(self):
        # websockets>=15: Server.close() is async; older versions return a
        # coroutine or run synchronously. Handle all shapes.
        if self._ws is not None:
            result = self._ws.close()
            if inspect.isawaitable(result):
                await result
        # NOTE: do NOT stop the loop here — that would interrupt
        # run_until_complete(_serve_ws) before the serve future settles.
        # _serve_ws polls _stopping and stops the loop itself.

    def _loop_run(self):
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._serve_ws())
        except Exception as exc:
            self.loop_exceptions.append(exc)
        finally:
            self.loop.close()

    # -- HTTP ---------------------------------------------------------------
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

        import socketserver
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._http_server = server
        self.http_url = f"http://127.0.0.1:{server.server_address[1]}"
        server.serve_forever()

    # -- websocket ----------------------------------------------------------
    async def _serve_ws(self):
        import socket

        import websockets

        # Separate port from the HTTP server (the two cannot share one).
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
                # The mock "browser process" survives client disconnects.
                pass

        self.ws_url = f"ws://127.0.0.1:{ws_port}/devtools/browser/mock"
        self._ws = await websockets.serve(handler, "127.0.0.1", ws_port)
        self._ready.set()

        # Poll until stop() signals shutdown (or the server closes naturally).
        # Stopping the loop from HERE (inside the serve future) keeps
        # run_until_complete happy — stopping it from a side callback would
        # raise "Event loop stopped before Future completed".
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
    """Scratch CDP endpoint: 1 stale page, 1 protected page, 1 blank page,
    1 iframe, 1 worker."""
    server = MockCdpServer([
        _target("stale-1", "https://stale.example/page1"),
        _target("stale-2", "https://stale.example/page2"),
        _target("active-1", "https://active.example/dash"),
        _target("blank-1", "about:blank"),
        _target("iframe-1", "https://active.example/dash", "iframe"),
        _target("worker-1", "blob:https://active.example", "worker"),
    ])
    server.start()
    yield server
    server.stop()


_RESTORED_ACTIVE = {}
_RESTORED_ACTIVITY = {}
_RESTORED_SWEEP_FLAG = False
_RESTORED_CLEANUP_RUNNING = False
_RESTORED_THREAD = None


def _reset_age_cache():
    browser_tool._page_ages_cache = {}
    browser_tool._page_ages_cache_loaded = False


def _restore_globals():
    browser_tool._active_sessions.clear()
    browser_tool._active_sessions.update(_RESTORED_ACTIVE)
    browser_tool._session_last_activity.clear()
    browser_tool._session_last_activity.update(_RESTORED_ACTIVITY)
    browser_tool._cdp_startup_sweep_done = _RESTORED_SWEEP_FLAG
    browser_tool._cleanup_running = _RESTORED_CLEANUP_RUNNING
    browser_tool._cleanup_thread = _RESTORED_THREAD
    _reset_age_cache()


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    global _RESTORED_ACTIVE, _RESTORED_ACTIVITY, _RESTORED_SWEEP_FLAG
    global _RESTORED_CLEANUP_RUNNING, _RESTORED_THREAD
    # Point the persistent page-age map at a scratch file so these tests never
    # read/write the real ~/.hermes/browser-page-ages.json, and start with an
    # empty (tracked) age map: in the new policy, unknown-age pages are never
    # TTL-reaped and unknown-age = newest for max-tabs, so a clean map means
    # "nothing is over-TTL, keep the newest N".
    monkeypatch.setenv("HERMES_BROWSER_AGE_FILE", str(tmp_path / "browser-page-ages.json"))
    _reset_age_cache()
    _RESTORED_ACTIVE = dict(browser_tool._active_sessions)
    _RESTORED_ACTIVITY = dict(browser_tool._session_last_activity)
    _RESTORED_SWEEP_FLAG = browser_tool._cdp_startup_sweep_done
    _RESTORED_CLEANUP_RUNNING = browser_tool._cleanup_running
    _RESTORED_THREAD = browser_tool._cleanup_thread
    yield
    _restore_globals()


# ============================================================================
# _close_stale_cdp_targets
# ============================================================================

class TestCloseStaleCdpTargets:
    def test_closes_stale_pages_only_and_endpoint_survives(self, mock_cdp):
        # No protect list: every page with a real URL is stale (blank page,
        # iframe, worker are never touched).
        summary = browser_tool._close_stale_cdp_targets(mock_cdp.ws_url)

        assert summary["total"] == 6
        assert summary["stale"] == 3
        assert summary["closed"] == 3
        assert summary["failed"] == 0
        assert summary["errors"] == []
        # All real pages gone (assert by target id — the surviving iframe
        # shares the active page's URL, so a URL-set check would be ambiguous).
        remaining = {t["id"] for t in mock_cdp.targets}
        assert remaining == {"blank-1", "iframe-1", "worker-1"}
        assert mock_cdp.closed_ids == ["stale-1", "stale-2", "active-1"]
        # The endpoint (mock "browser process") survives the closes.
        assert mock_cdp.connections >= 1
        assert not mock_cdp.loop_exceptions

    def test_protected_url_kept_exact_and_prefix(self, mock_cdp):
        summary = browser_tool._close_stale_cdp_targets(
            mock_cdp.ws_url,
            protect_urls=("https://active.example",),  # prefix match
        )
        assert summary["stale"] == 2
        assert summary["closed"] == 2
        assert summary["failed"] == 0
        assert "https://active.example/dash" in {t["url"] for t in mock_cdp.targets}
        assert mock_cdp.closed_ids == ["stale-1", "stale-2"]

    def test_dead_endpoint_is_swallowed(self):
        # Nothing listening on 127.0.0.1:1 — must not raise, must report failure.
        summary = browser_tool._close_stale_cdp_targets(
            "ws://127.0.0.1:1/devtools/browser/dead", timeout=2.0,
        )
        assert summary["closed"] == 0
        assert summary["errors"], "expected a recorded failure for a dead endpoint"

    def test_empty_ws_url_noop(self):
        summary = browser_tool._close_stale_cdp_targets("")
        assert summary == {
            "ws_url": "", "total": 0, "stale": 0, "closed": 0, "failed": 0,
            "errors": [],
        }


class TestIsStaleCdpTarget:
    def test_only_real_pages_are_stale(self):
        assert browser_tool._is_stale_cdp_target(_target("a", "https://x/"))
        assert not browser_tool._is_stale_cdp_target(_target("a", "about:blank"))
        assert not browser_tool._is_stale_cdp_target(_target("a", ""))
        assert not browser_tool._is_stale_cdp_target(_target("a", "https://x/", "iframe"))
        assert not browser_tool._is_stale_cdp_target(_target("a", "blob:https://x", "worker"))
        assert not browser_tool._is_stale_cdp_target(_target("a", "devtools://x", "browser_ui"))


class TestCdpWsUrlFor:
    def test_ws_passthrough(self):
        assert (
            browser_tool._cdp_ws_url_for("ws://h:9222/devtools/browser/abc")
            == "ws://h:9222/devtools/browser/abc"
        )

    def test_http_discovery(self, mock_cdp):
        assert (
            browser_tool._cdp_ws_url_for(mock_cdp.http_url)
            == mock_cdp.ws_url
        )

    def test_dead_http_discovery_returns_empty(self):
        assert browser_tool._cdp_ws_url_for("http://127.0.0.1:1") == ""

    def test_empty_and_bare_schemes(self):
        assert browser_tool._cdp_ws_url_for("") == ""
        assert browser_tool._cdp_ws_url_for("not a url") == ""


# ============================================================================
# _cleanup_single_browser_session — raw-CDP close path for cdp_url sessions
# ============================================================================

class TestCleanupSingleCdpSession:
    def _seed_ages(self, now):
        """Pre-seed the persistent age map so the stale pages are over-TTL and
        the active page is fresh (TTL = default 86400s)."""
        browser_tool._update_page_ages(
            [
                "https://stale.example/page1",
                "https://stale.example/page2",
                "https://active.example/dash",
            ],
            now=now - 100000,  # record first-seen in the past
        )
        # Overwrite the active page to look fresh.
        ages = browser_tool._read_page_ages()
        ages["https://active.example/dash"] = now - 10
        browser_tool._write_page_ages(ages)

    def test_cdp_session_closes_stale_targets_via_raw_cdp(self, mock_cdp):
        now = time_now()
        self._seed_ages(now)
        browser_tool._active_sessions["task-cdp"] = {
            "session_name": "cdp_mock",
            "bb_session_id": None,
            "cdp_url": mock_cdp.ws_url,
            "features": {"cdp_override": True},
        }
        # INACTIVE session (activity entry present) -> the reaper protects
        # nothing, so over-TTL pages (including any the session last showed)
        # become normal candidates.
        browser_tool._session_last_activity["task-cdp"] = now

        with unittest.mock.patch("tools.browser_tool._run_browser_command") as mock_run:
            browser_tool._cleanup_single_browser_session("task-cdp")

        # The hanging/no-op agent-browser close must NOT run for cdp_url
        # sessions; the raw-CDP path closes the over-TTL pages instead.
        mock_run.assert_not_called()
        # Bookkeeping removed.
        assert "task-cdp" not in browser_tool._active_sessions
        assert "task-cdp" not in browser_tool._session_last_activity
        # Over-TTL pages closed; the fresh active page + blank + iframe survive
        # (assert by id — the iframe shares the active page's URL).
        remaining = {t["id"] for t in mock_cdp.targets}
        assert remaining == {"active-1", "blank-1", "iframe-1", "worker-1"}
        assert sorted(mock_cdp.closed_ids) == ["stale-1", "stale-2"]
        assert mock_cdp.connections >= 1
        assert not mock_cdp.loop_exceptions

    def test_cdp_session_expiry_replacement_protects_its_last_url(self, mock_cdp):
        now = time_now()
        # Seed ALL pages (incl. the active one) over-TTL so that ONLY the
        # protection (protect_last_url=True) can save the active page — a
        # faithful test of the expiry-replacement protection, not of TTL.
        browser_tool._update_page_ages(
            [
                "https://stale.example/page1",
                "https://stale.example/page2",
                "https://active.example/dash",
            ],
            now=now - 100000,
        )
        browser_tool._active_sessions["task-cdp"] = {
            "session_name": "cdp_mock",
            "bb_session_id": None,
            "cdp_url": mock_cdp.ws_url,
            "features": {"cdp_override": True},
            "last_url": "https://active.example/dash",
        }
        # EXPIRY-REPLACEMENT path: the session is being renewed, so the
        # in-use last_url must be protected from the reaping plan.
        assert "task-cdp" not in browser_tool._session_last_activity

        with unittest.mock.patch("tools.browser_tool._run_browser_command") as mock_run:
            browser_tool._cleanup_single_browser_session("task-cdp", protect_last_url=True)

        mock_run.assert_not_called()
        # The protected last_url (active) survives even though it is over-TTL;
        # the unprotected over-TTL stale pages are closed.
        urls = {t["url"] for t in mock_cdp.targets}
        assert "https://active.example/dash" in urls
        assert sorted(mock_cdp.closed_ids) == ["stale-1", "stale-2"]

    def test_non_cdp_session_still_uses_agent_browser_close(self):
        browser_tool._active_sessions["task-local"] = {
            "session_name": "local_mock",
            "bb_session_id": None,
        }
        browser_tool._session_last_activity["task-local"] = time_now()

        with unittest.mock.patch(
            "tools.browser_tool._run_browser_command",
            return_value={"success": True},
        ) as mock_run:
            browser_tool._cleanup_single_browser_session("task-local")

        mock_run.assert_called_once_with("task-local", "close", [], timeout=10)
        assert "task-local" not in browser_tool._active_sessions


def time_now():
    import time
    return time.time()


# ============================================================================
# _sweep_stale_cdp_targets — startup sweep
# ============================================================================

class TestSweepStaleCdpTargets:
    def _seed_ages(self, now):
        """Pre-seed the persistent age map: stale pages over-TTL, active fresh."""
        browser_tool._update_page_ages(
            [
                "https://stale.example/page1",
                "https://stale.example/page2",
                "https://active.example/dash",
            ],
            now=now - 100000,
        )
        ages = browser_tool._read_page_ages()
        ages["https://active.example/dash"] = now - 10
        browser_tool._write_page_ages(ages)

    def test_sweep_closes_stale_and_protects_active_session(self, mock_cdp, monkeypatch):
        monkeypatch.setenv("BROWSER_CDP_URL", mock_cdp.http_url)
        self._seed_ages(time_now())
        # Active session in this process (the very one that started the
        # cleanup thread) — its live page must survive the sweep.
        browser_tool._active_sessions["task-sweep"] = {
            "session_name": "cdp_sweep",
            "bb_session_id": None,
            "cdp_url": mock_cdp.ws_url,
            "last_url": "https://active.example/dash",
        }

        with unittest.mock.patch("tools.browser_tool.time.sleep"):
            browser_tool._sweep_stale_cdp_targets()

        urls = {t["url"] for t in mock_cdp.targets}
        assert "https://stale.example/page1" not in urls
        assert "https://stale.example/page2" not in urls
        assert "https://active.example/dash" in urls
        assert "about:blank" in urls
        assert sorted(mock_cdp.closed_ids) == ["stale-1", "stale-2"]

    def test_sweep_no_op_without_cdp_override(self, monkeypatch):
        monkeypatch.delenv("BROWSER_CDP_URL", raising=False)

        with unittest.mock.patch("tools.browser_tool._cdp_ws_url_for") as mock_ws:
            browser_tool._sweep_stale_cdp_targets()
        mock_ws.assert_not_called()

    def test_sweep_dead_endpoint_is_swallowed(self, monkeypatch):
        monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:1")

        with unittest.mock.patch("tools.browser_tool.time.sleep"):
            browser_tool._sweep_stale_cdp_targets()  # must not raise

    def test_cleanup_thread_runs_sweep_once_on_startup(self, mock_cdp, monkeypatch):
        monkeypatch.setenv("BROWSER_CDP_URL", mock_cdp.http_url)
        self._seed_ages(time_now())
        browser_tool._cdp_startup_sweep_done = False
        browser_tool._cleanup_running = True
        # The active session's page is recorded as last_url — the sweep must
        # protect exactly that page.
        browser_tool._active_sessions["task-sweep"] = {
            "session_name": "cdp_sweep",
            "bb_session_id": None,
            "cdp_url": mock_cdp.ws_url,
            "last_url": "https://active.example/dash",
        }

        # Deterministic: mock the sleep (no real waits), patch the orphan
        # reaper (no real filesystem scan), and drop _cleanup_running on the
        # first 1s loop tick so the 30x1s loop exits on its first pass.
        def _sleep_effect(seconds):
            if seconds == 1:
                browser_tool._cleanup_running = False

        mock_sleep = unittest.mock.Mock(side_effect=_sleep_effect)
        with unittest.mock.patch("tools.browser_tool.time.sleep", mock_sleep), \
                unittest.mock.patch("tools.browser_tool._reap_orphaned_browser_sessions"):
            browser_tool._browser_cleanup_thread_worker()

        # One-time sweep ran with its 5 s grace sleep; stale pages closed.
        assert any(c.args and c.args[0] == 5 for c in mock_sleep.call_args_list)
        assert sorted(mock_cdp.closed_ids) == ["stale-1", "stale-2"]
        assert browser_tool._cdp_startup_sweep_done is True

        # Second invocation must not re-sweep (flag already set).
        mock_cdp.targets.extend([
            _target("stale-3", "https://stale.example/page3"),
        ])
        browser_tool._cleanup_running = True
        with unittest.mock.patch("tools.browser_tool.time.sleep", mock_sleep), \
                unittest.mock.patch("tools.browser_tool._reap_orphaned_browser_sessions"):
            browser_tool._browser_cleanup_thread_worker()
        assert sorted(mock_cdp.closed_ids) == ["stale-1", "stale-2"]
