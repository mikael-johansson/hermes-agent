"""Tests for the auto-reflect "one turn behind" feature of the Hindsight
memory provider (config key ``auto_reflect``, default off).

Covers the user-approved design:
- a background agentic reflect fires per non-trivial turn (max one in
  flight, skip-if-busy);
- its result is injected ONCE on the NEXT turn's prefetch() as a dedicated
  "# Hindsight Reflection" section, alongside (not instead of) the
  recall slot;
- trivial and system/delegation-notification queries are skipped;
- the in-flight reflect is cancelled and its slot cleared on session
  switch and shutdown;
- auto_reflect=False and tools mode never fire a reflect.

All tests use the mock client only (no real daemon, no network). The
real loop/future machinery is left in place: with an AsyncMock areflect
the coroutine completes almost immediately, so completion is
deterministic via a short join of the collector thread.
"""

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.memory.hindsight import HindsightMemoryProvider


# ---------------------------------------------------------------------------
# Fixtures (mirroring tests/plugins/memory/test_hindsight_provider.py)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(tmp_path, monkeypatch):
    """Ensure no stale env vars leak between tests."""
    for key in (
        "HINDSIGHT_API_KEY", "HINDSIGHT_API_URL", "HINDSIGHT_BANK_ID",
        "HINDSIGHT_BUDGET", "HINDSIGHT_MODE", "HINDSIGHT_TIMEOUT",
        "HINDSIGHT_IDLE_TIMEOUT", "HINDSIGHT_LLM_API_KEY",
        "HINDSIGHT_RETAIN_TAGS", "HINDSIGHT_RETAIN_OBSERVATION_SCOPES",
        "HINDSIGHT_RETAIN_SOURCE",
        "HINDSIGHT_RETAIN_USER_PREFIX", "HINDSIGHT_RETAIN_ASSISTANT_PREFIX",
    ):
        monkeypatch.delenv(key, raising=False)
    isolated_home = tmp_path / "user-home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: isolated_home))


def _make_mock_client():
    """Mock Hindsight client with async methods."""
    client = MagicMock()
    client.arecall = AsyncMock(
        return_value=SimpleNamespace(
            results=[SimpleNamespace(text="Memory 1")]
        )
    )
    client.areflect = AsyncMock(
        return_value=SimpleNamespace(text="Synthesized reflection text")
    )
    client.aretain_batch = AsyncMock()
    client.aclose = AsyncMock()
    return client


def _make_provider(tmp_path, monkeypatch, **overrides):
    """Create an initialized provider with a mock client and a fresh
    reflect slot, applying config overrides."""
    config = {
        "mode": "cloud",
        "apiKey": "test-key",
        "api_url": "http://localhost:9999",
        "bank_id": "test-bank",
        "budget": "mid",
        "memory_mode": "hybrid",
        "auto_reflect": True,
    }
    config.update(overrides)
    config_path = tmp_path / "hindsight" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config))

    monkeypatch.setattr(
        "plugins.memory.hindsight.get_hermes_home", lambda: tmp_path
    )

    p = HindsightMemoryProvider()
    p.initialize(session_id="test-session", hermes_home=str(tmp_path), platform="cli")
    p._client = _make_mock_client()
    # Initialize()'s config parsing is the production path for
    # self._auto_reflect; re-assert the fixture applied it so tests fail
    # loudly if the key is ever renamed.
    expected = bool(config.get("auto_reflect", False))
    assert p._auto_reflect is expected
    # Reset reflect slot to a pristine state (provider may have been
    # reused across monkeypatch resets in theory; belt and braces).
    with p._reflect_lock:
        p._reflect_result = ""
        p._reflect_future = None
        p._reflect_pending_query = ""
    p._reflect_thread = None
    return p


def _wait_for_reflect(p, timeout=5.0):
    """Join the collector thread so the reflect result has landed."""
    t = p._reflect_thread
    if t is not None and t.is_alive():
        t.join(timeout=timeout)


def _wait_until(cond, timeout=5.0):
    """Poll a condition from the test thread until true (or timeout)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def _wait_for_prefetch(p, timeout=5.0):
    t = p._prefetch_thread
    if t is not None and t.is_alive():
        t.join(timeout=timeout)


# ---------------------------------------------------------------------------
# Config / gating
# ---------------------------------------------------------------------------


class TestAutoReflectConfig:
    def test_auto_reflect_defaults_off(self, tmp_path, monkeypatch):
        """The new config key defaults to False: existing users are
        unaffected until they opt in."""
        config = {
            "mode": "cloud",
            "apiKey": "test-key",
            "api_url": "http://localhost:9999",
            "bank_id": "test-bank",
            "budget": "mid",
            "memory_mode": "hybrid",
        }
        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home", lambda: tmp_path
        )
        p = HindsightMemoryProvider()
        p.initialize(session_id="s", hermes_home=str(tmp_path), platform="cli")
        p._client = _make_mock_client()
        assert p._auto_reflect is False

    def test_auto_reflect_parsed_from_config(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch, auto_reflect=True)
        assert p._auto_reflect is True

    def test_config_schema_contains_auto_reflect(self):
        """The new key is advertised in the config schema."""
        schema = HindsightMemoryProvider().get_config_schema()
        keys = [entry.get("key") for entry in schema]
        assert "auto_reflect" in keys
        entry = next(e for e in schema if e.get("key") == "auto_reflect")
        assert entry.get("default") is False

    def test_system_prompt_block_mentions_reflection_when_enabled(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch, auto_reflect=True)
        block = p.system_prompt_block()
        assert "# Hindsight Reflection" in block
        assert "one turn after each substantial question" in block

    def test_system_prompt_block_unchanged_when_disabled(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch, auto_reflect=False)
        block = p.system_prompt_block()
        assert "# Hindsight Reflection" not in block
        # The original hybrid-mode wording is intact.
        assert "Relevant memories are automatically injected into context." in block
        assert "Use hindsight_recall to search" in block

    def test_system_prompt_tools_mode_with_auto_reflect(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch, memory_mode="tools", auto_reflect=True)
        block = p.system_prompt_block()
        assert "tools mode" in block
        assert "# Hindsight Reflection" in block


class TestAutoReflectGating:
    def test_disabled_never_fires(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch, auto_reflect=False)
        p.queue_prefetch("a substantial enough query to pass")
        _wait_for_prefetch(p)
        _wait_for_reflect(p)
        assert p._client.areflect.call_count == 0
        assert p._reflect_future is None
        # No reflection section may ever appear; the (separate) recall
        # slot keeps its pre-existing behavior and may still be populated.
        out = p.prefetch("next")
        assert "# Hindsight Reflection" not in out
        assert "Synthesized reflection text" not in out

    def test_tools_mode_never_fires(self, tmp_path, monkeypatch):
        """tools mode skips the recall path entirely, and the reflect is
        skipped too (it hangs off queue_prefetch, which returns early)."""
        p = _make_provider(tmp_path, monkeypatch, memory_mode="tools", auto_reflect=True)
        p.queue_prefetch("a substantial enough query to pass")
        _wait_for_prefetch(p)
        _wait_for_reflect(p)
        assert p._client.areflect.call_count == 0

    def test_empty_query_skips(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch)
        p.queue_auto_reflect("")
        _wait_for_reflect(p)
        assert p._client.areflect.call_count == 0

    def test_whitespace_only_query_skips(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch)
        p.queue_auto_reflect("   ")
        _wait_for_reflect(p)
        assert p._client.areflect.call_count == 0

    def test_trivial_short_query_skips(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch)
        p.queue_auto_reflect("ok")
        _wait_for_reflect(p)
        assert p._client.areflect.call_count == 0

    def test_notification_query_skipped(self, tmp_path, monkeypatch):
        """System/delegation/batch notifications (leading '[') must not
        trigger an expensive background synthesis."""
        p = _make_provider(tmp_path, monkeypatch)
        p.queue_auto_reflect("[agent finished: subagent-2 completed its task]")
        _wait_for_reflect(p)
        assert p._client.areflect.call_count == 0

    def test_shutting_down_skips(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch)
        p._shutting_down.set()
        p.queue_auto_reflect("a substantial enough query to pass")
        _wait_for_reflect(p)
        assert p._client.areflect.call_count == 0

    def test_non_trivial_query_fires(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch)
        p.queue_auto_reflect("What decisions did we make about the GPU build?")
        _wait_for_reflect(p)
        assert p._client.areflect.call_count == 1
        assert p._reflect_future is None  # released after completion

    def test_queue_prefetch_triggers_auto_reflect(self, tmp_path, monkeypatch):
        """queue_prefetch must fire the reflect after its existing logic —
        the recall path and the reflect are separate axes."""
        p = _make_provider(tmp_path, monkeypatch)
        p.queue_prefetch("What did we decide about the GPU build yesterday?")
        _wait_for_prefetch(p)
        _wait_for_reflect(p)
        # Both paths ran: recall for the fast slot, reflect for next turn.
        assert p._client.arecall.call_count == 1
        assert p._client.areflect.call_count == 1
        # The recall slot kept its existing behavior and got its result.
        assert "Memory 1" in p.prefetch("next")


# ---------------------------------------------------------------------------
# Next-turn consumption
# ---------------------------------------------------------------------------


class TestReflectionInjection:
    def test_reflected_on_next_turn(self, tmp_path, monkeypatch):
        """Fired on turn N; consumed on turn N+1's prefetch as a dedicated
        '# Hindsight Reflection' section."""
        p = _make_provider(tmp_path, monkeypatch)
        p.queue_prefetch("What did we decide about the GPU build yesterday?")
        # Same turn's prefetch (turn N): the reflect was just fired, it
        # hasn't completed in any meaningful sense — but with the mock it
        # is effectively instant, so allow it to land: the KEY contract is
        # that a LATER turn consumes it once.
        _wait_for_reflect(p)
        first = p.prefetch("same turn query")
        assert "Synthesized reflection text" in first
        assert "# Hindsight Reflection" in first

    def test_consumed_once_not_twice(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch)
        p.queue_prefetch("What did we decide about the GPU build yesterday?")
        _wait_for_reflect(p)
        first = p.prefetch("turn N+1")
        assert "# Hindsight Reflection" in first
        second = p.prefetch("turn N+2")
        assert "# Hindsight Reflection" not in second
        assert "Synthesized reflection text" not in second

    def test_reflection_without_recall_returns_reflection_only(self, tmp_path, monkeypatch):
        """If the recall slot is empty but a reflection is pending, the
        overall contract ('' if nothing) still holds and the reflection
        section is returned on its own."""
        p = _make_provider(tmp_path, monkeypatch)
        p._client.arecall.return_value = SimpleNamespace(results=[])
        p.queue_prefetch("What did we decide about the GPU build yesterday?")
        _wait_for_prefetch(p)
        _wait_for_reflect(p)
        out = p.prefetch("next")
        assert out != ""
        assert "# Hindsight Reflection" in out
        assert "Synthesized reflection text" in out
        # No recall section (recall was empty).
        assert "# Hindsight Memory (persistent cross-session context)" not in out

    def test_reflection_and_recall_injected_together(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch)
        p.queue_prefetch("What did we decide about the GPU build yesterday?")
        _wait_for_prefetch(p)
        _wait_for_reflect(p)
        out = p.prefetch("next")
        assert "# Hindsight Memory (persistent cross-session context)" in out
        assert "Memory 1" in out
        assert "# Hindsight Reflection" in out
        assert "Synthesized reflection text" in out

    def test_empty_reflect_result_injects_nothing(self, tmp_path, monkeypatch):
        """A reflect that returns no text stores nothing: nothing is
        injected on the next turn (and the slot stays clear)."""
        p = _make_provider(tmp_path, monkeypatch)
        p._client.areflect.return_value = SimpleNamespace(text="")
        p.queue_auto_reflect("What did we decide about the GPU build yesterday?")
        _wait_for_reflect(p)
        assert p._reflect_result == ""
        assert p.prefetch("next") == ""

    def test_empty_reflect_result_released_and_next_turn_can_reflect(
        self, tmp_path, monkeypatch
    ):
        p = _make_provider(tmp_path, monkeypatch)
        p._client.areflect.return_value = SimpleNamespace(text="")
        p.queue_auto_reflect("What did we decide about the GPU build yesterday?")
        _wait_for_reflect(p)
        # Slot released, so the next non-trivial turn can start a fresh
        # reflect (no permanent "busy" state from a no-op reflect).
        p._client.areflect.return_value = SimpleNamespace(text="Fresh text")
        p.queue_auto_reflect("Follow-up question about the same build decision")
        _wait_for_reflect(p)
        assert "Fresh text" in p.prefetch("next")


# ---------------------------------------------------------------------------
# Skip-if-busy (max one in flight)
# ---------------------------------------------------------------------------


class TestSkipIfBusy:
    def test_second_turn_while_first_pending_does_not_reflect_again(
        self, tmp_path, monkeypatch
    ):
        """While a reflect is in flight, a new turn must NOT start a second
        one (no orphan pileup). The mock areflect blocks (via
        asyncio.to_thread — an executor thread, so the shared event loop
        stays free) until the gate is released, keeping the first future
        pending across the second queue_prefetch."""
        p = _make_provider(tmp_path, monkeypatch)
        box: dict = {}

        async def _gated_areflect(bank_id, query, budget):
            # An asyncio.Event gate: the awaited waiter is a loop-level
            # future, which IS prompt-cancellable (unlike to_thread's
            # executor future) — and it blocks only the task, never the
            # loop, so the cancel callback stays runnable.
            box["gate"] = asyncio.Event()
            await box["gate"].wait()
            return SimpleNamespace(text="Gated reflection")

        p._client.areflect = AsyncMock(side_effect=_gated_areflect)

        p.queue_prefetch("First substantial question about the project plan?")
        started = time.monotonic()
        while "gate" not in box and time.monotonic() - started < 5.0:
            time.sleep(0.01)
        assert "gate" in box, "areflect coroutine never started"
        assert p._client.areflect.call_count == 1
        assert p._reflect_future is not None
        assert not p._reflect_future.done()

        # Second turn arrives while the first reflect is still pending.
        p.queue_prefetch("Second substantial question about the same plan?")
        _wait_for_prefetch(p)

        assert p._client.areflect.call_count == 1, (
            "skip-if-busy violated: a second reflect was started while the "
            "first was still in flight"
        )

        # Release the gate (thread-safely, on the shared loop); the single
        # reflect completes and its result is consumed once on the
        # following prefetch, alongside the (fast) recall section.
        from plugins.memory.hindsight import _get_loop
        _get_loop().call_soon_threadsafe(box["gate"].set)
        _wait_for_reflect(p)
        out = p.prefetch("third turn")
        assert "# Hindsight Memory (persistent cross-session context)" in out
        assert "# Hindsight Reflection" in out
        assert "Gated reflection" in out
        # Consume-once for both slots: the turn after has neither.
        assert p.prefetch("fourth turn") == ""

    def test_completed_future_does_not_block_new_reflect(self, tmp_path, monkeypatch):
        """Once a reflect completes (future done), the next non-trivial
        turn starts a fresh one — the busy check must only apply while
        in flight."""
        p = _make_provider(tmp_path, monkeypatch)
        p.queue_auto_reflect("First question about a project decision we made?")
        _wait_for_reflect(p)
        assert p._client.areflect.call_count == 1
        assert p._reflect_future is None
        p.queue_auto_reflect("Second question about a different project area")
        _wait_for_reflect(p)
        assert p._client.areflect.call_count == 2


# ---------------------------------------------------------------------------
# Cancellation on session switch / shutdown
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_session_switch_cancels_inflight_and_clears_slot(
        self, tmp_path, monkeypatch
    ):
        """A reflect still in flight at session switch time is cancelled
        (server-side request abort) and its slot cleared, so the new
        session's first turn gets nothing from the old session."""
        p = _make_provider(tmp_path, monkeypatch)
        box: dict = {}
        cancelled = []

        async def _gated_areflect(bank_id, query, budget):
            # asyncio.Event gate: once the task yields on gate.wait() it is
            # a loop-level waiter — prompt-cancellable (unlike to_thread's
            # executor future), and it never blocks the loop itself.
            box["gate"] = asyncio.Event()
            try:
                await box["gate"].wait()
                return SimpleNamespace(text="Stale reflection")
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        p._client.areflect = AsyncMock(side_effect=_gated_areflect)
        p.queue_prefetch("A substantial question from the old session today")
        started = time.monotonic()
        while "gate" not in box and time.monotonic() - started < 5.0:
            time.sleep(0.01)
        assert "gate" in box, "areflect coroutine never started"
        assert p._client.areflect.call_count == 1
        assert p._reflect_future is not None
        assert not p._reflect_future.done()

        p.on_session_switch("new-session-id")

        # The switch joins the collector (<=1s), but the loop-level cancel
        # callback it posted may still be pending when the switch returns;
        # wait for the coroutine to actually observe the cancellation
        # before asserting the rest.
        assert _wait_until(lambda: bool(cancelled)), (
            "the in-flight reflect was not cancelled"
        )
        _wait_for_reflect(p)
        assert p._reflect_future is None
        assert p._reflect_result == ""
        assert p._reflect_thread is None

        # The new session's first prefetch must not get the old session's
        # reflection. (It may still get the fast recall for the old
        # session's query — that slot is cleared by the switch too, but
        # the mock recall re-populates it for this very call; the
        # reflection slot is what must stay empty.)
        first = p.prefetch("first turn of new session")
        assert "# Hindsight Reflection" not in first
        assert "Stale reflection" not in first

        # Belt and braces: release the (already cancelled) gate
        # thread-safely; even if anything completed late, the collector
        # only stores when its future is still the current one (None now),
        # so no leak into the new session.
        from plugins.memory.hindsight import _get_loop
        _get_loop().call_soon_threadsafe(box["gate"].set)
        time.sleep(0.2)
        assert p._reflect_result == ""
        assert "# Hindsight Reflection" not in p.prefetch("second turn of new session")

    def test_shutdown_cancels_inflight_and_clears_slot(self, tmp_path, monkeypatch):
        p = _make_provider(tmp_path, monkeypatch)
        box: dict = {}

        async def _gated_areflect(bank_id, query, budget):
            box["gate"] = asyncio.Event()
            try:
                await box["gate"].wait()
                return SimpleNamespace(text="Should never store")
            except asyncio.CancelledError:
                raise

        p._client.areflect = AsyncMock(side_effect=_gated_areflect)
        p.queue_prefetch("A substantial question before shutdown happens now")
        started = time.monotonic()
        while "gate" not in box and time.monotonic() - started < 5.0:
            time.sleep(0.01)
        assert "gate" in box
        assert p._reflect_future is not None

        p.shutdown()

        assert p._shutting_down.is_set()
        assert p._reflect_future is None
        assert p._reflect_result == ""
        # The reflect slot must be dead; a post-shutdown prefetch must not
        # surface a reflection. (The recall slot is intentionally NOT
        # cleared by shutdown() — pre-existing behavior — so its stale
        # mock content may still be returned, but never a reflection.)
        out = p.prefetch("post-shutdown")
        assert "# Hindsight Reflection" not in out
        assert "Should never store" not in out
        from plugins.memory.hindsight import _get_loop
        _get_loop().call_soon_threadsafe(box["gate"].set)

    def test_shutdown_with_stored_result_still_clears(self, tmp_path, monkeypatch):
        """Even a COMPLETED (stored) reflection is cleared on shutdown:
        unconditionally, per spec."""
        p = _make_provider(tmp_path, monkeypatch)
        p.queue_auto_reflect("A substantial question that will complete fast")
        _wait_for_reflect(p)
        assert p._reflect_result == "Synthesized reflection text"
        p.shutdown()
        assert p._reflect_result == ""
        assert p.prefetch("x") == ""


# ---------------------------------------------------------------------------
# Recall slot regression guards (existing behavior must be intact)
# ---------------------------------------------------------------------------


class TestRecallSlotUnchanged:
    def test_prefetch_consume_once_still_holds(self, tmp_path, monkeypatch):
        """The recall slot's consume-once semantics are unchanged by the
        new reflect slot: with auto_reflect on, the recall result is still
        consumed exactly once, independent of the reflection."""
        p = _make_provider(tmp_path, monkeypatch)
        p.queue_prefetch("A substantial question about a prior decision we made")
        _wait_for_prefetch(p)
        first = p.prefetch("turn N+1")
        assert "Memory 1" in first
        second = p.prefetch("turn N+2")
        assert "Memory 1" not in second

    def test_prefetch_returns_empty_when_both_slots_empty(
        self, tmp_path, monkeypatch
    ):
        p = _make_provider(tmp_path, monkeypatch)
        assert p.prefetch("anything") == ""

    def test_prefetch_still_joins_background_thread(self, tmp_path, monkeypatch):
        """prefetch() must still join the recall thread (3s cap) before
        reading the slot — the reflect additions must not have broken the
        reply-path join."""
        import threading as _t

        p = _make_provider(tmp_path, monkeypatch)
        gate = _t.Event()
        finished = _t.Event()

        def _slow_recall():
            gate.wait(timeout=5.0)
            with p._prefetch_lock:
                p._prefetch_result = "late recall"
            finished.set()

        p._prefetch_thread = _t.Thread(target=_slow_recall, daemon=True)
        p._prefetch_thread.start()
        gate.set()
        out = p.prefetch("q")
        assert finished.is_set(), "prefetch returned before joining the thread"
        assert "late recall" in out
