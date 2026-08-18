"""Pre-send fit guard: compress BEFORE sending into a hard context wall.

The compression threshold is a soft "compact for health" trigger that may sit
well below the model's window (e.g. 57% of 150K), and a rough estimate can
undercount a dense payload even after per-model calibration. The gap between
the *threshold* and the provider's hard wall is the zone where a session
sends into the wall with no pre-send signal — for a hard-wall provider
(llama.cpp) that is a truncated stream and a re-send loop, not a clean
context-overflow error.

``prompt_fits_window()`` is wired into the pre-API pressure gate as a second
trigger (``_over_threshold or _pre_send_wall``) for the SAME pre-API
compression pass — one mechanism, no parallel block: when the calibrated
request estimate leaves no room for a response inside the known window,
compress now instead of sending into the wall.

The guard's unique zone is ``usable_input < est < threshold`` — i.e. the
reserved output budget (``max_tokens``) or the threshold itself pushes the
usable input budget below the trigger line. With ``max_tokens=None`` the
usable budget is ``window - 512``, which sits ABOVE any sane threshold, so
the threshold trigger covers the gap; the guard adds protection when a large
output reservation shrinks the usable budget below the threshold.

These tests pin the loop wiring:

* a request in the guard's zone (below threshold, above usable budget)
  compresses pre-send, then the compacted request is sent,
* a request that fits sends without a compression pass (no false fire),
* the guard reuses the pre-API pass's defer predicate and stands down behind
  it, and stays fail-open for an unknown window,
* the (rough, real) note fed to the defer projection is the calibrated basis
  (the loop's estimate carries the per-model scale).

Unit coverage for ``prompt_fits_window`` / ``apply_token_estimate_scale`` /
``resolve_token_estimate_scale`` lives in
tests/agent/test_token_estimate_scale.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
import run_agent


# ---------------------------------------------------------------------------
# Fixture (mirrors tests/run_agent/test_compression_lock_defer.py)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_compression_sleep(monkeypatch):
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0)


def _mock_response(content="ok", finish_reason="stop"):
    msg = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = None
    return resp


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        a.compression_enabled = True
        a.save_trajectories = False
        return a


def _wall_compressor(window: int, threshold: int) -> SimpleNamespace:
    """A hard-wall provider's compressor: known window, threshold BELOW it.

    ``threshold_tokens`` is what the user set (e.g. 57% of the window); the
    guard under test is the region between the usable input budget and the
    wall.
    """
    return SimpleNamespace(
        protect_first_n=3,
        protect_last_n=20,
        threshold_tokens=threshold,
        context_length=window,
        last_prompt_tokens=0,
        should_compress=lambda t: t >= threshold,
        should_defer_preflight_to_real_usage=lambda _t: False,
        get_active_compression_failure_cooldown=lambda: None,
    )


def _patch_estimates(est: int):
    """Pin the request estimate on BOTH module namespaces.

    ``build_turn_context`` (prologue preflight) and the pre-API gate each
    import ``estimate_*`` into their own namespace; patching both keeps the
    prologue and the guard on the same basis. (With a 3-message history the
    prologue's cheap gate stands down, so only the loop's estimate is ever
    consumed here.)

    The mock is *scale-aware*: it returns ``apply_token_estimate_scale(est,
    scale)``, honoring the per-model calibration exactly the way the real
    estimators do (they apply the scale to the aggregate total). At the
    default scale 1.0 this is byte-identical to a plain ``return_value=est``,
    so the guard tests are unaffected; with a calibrated scale it returns the
    calibrated figure — the basis the (rough, real) note and the fit guard
    are supposed to see.
    """
    from agent.model_metadata import apply_token_estimate_scale

    def _msgs(_messages, scale=1.0):
        return apply_token_estimate_scale(est, scale)

    def _req(_messages, *, system_prompt="", tools=None, scale=1.0):
        return apply_token_estimate_scale(est, scale)

    return (
        patch("agent.turn_context.estimate_request_tokens_rough", side_effect=_req),
        patch("agent.turn_context.estimate_messages_tokens_rough", side_effect=_msgs),
        patch("agent.conversation_loop.estimate_request_tokens_rough", side_effect=_req),
        patch("agent.conversation_loop.estimate_messages_tokens_rough", side_effect=_msgs),
    )


_PREFILL = [
    {"role": "user", "content": "previous question"},
    {"role": "assistant", "content": "previous answer"},
]


# ---------------------------------------------------------------------------
# The guard: request in the unique zone compresses pre-send
# ---------------------------------------------------------------------------


class TestPreSendFitGuard:
    def test_guard_compresses_then_sends_fitting_request(self, agent):
        """Window 150,000, threshold 85,500 (57%), output reservation
        ``max_tokens`` 70,000 → usable input budget 80,000, which is BELOW
        the threshold. Calibrated estimate 82,000 sits in the guard's unique
        zone: below the soft threshold (so the threshold trigger would let
        this request through) but above the usable budget (sending it would
        hit the wall).

        The pre-send pass must (a) fire, (b) re-run the gate on the
        compressed request (estimate 30,000, which fits), and (c) send it —
        with exactly ONE provider call, the compacted request.
        """
        agent.context_compressor = _wall_compressor(150_000, 85_500)
        agent.max_tokens = 70_000  # reserved output shrinks usable to 80,000
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="answered after pre-send compact"),
        ]

        # Two-phase estimate: 82,000 (guard zone) before the pass, 30,000
        # (fits) after. The loop re-estimates the rebuilt request on the
        # `continue` back to the top.
        est_values = iter([82_000, 30_000, 30_000])

        def _est(*_a, **_k):
            return next(est_values, 30_000)

        compressed = [{"role": "user", "content": "compact summary"}]
        compress_calls = []

        def _compress(messages, _system_message, **_kwargs):
            compress_calls.append(len(messages))
            return compressed, "You are helpful."

        with (
            patch("agent.turn_context.estimate_request_tokens_rough", return_value=30_000),
            patch("agent.turn_context.estimate_messages_tokens_rough", return_value=30_000),
            patch("agent.conversation_loop.estimate_request_tokens_rough", side_effect=_est),
            patch("agent.conversation_loop.estimate_messages_tokens_rough", side_effect=_est),
            patch.object(agent, "_compress_context", side_effect=_compress),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                "hello", conversation_history=list(_PREFILL)
            )

        # The guard fired: one compression pass, before the (only) API call.
        assert len(compress_calls) == 1, (
            f"guard did not fire pre-send: compress called {len(compress_calls)}x"
        )
        assert result.get("completed") is True
        assert result["final_response"] == "answered after pre-send compact"
        # Exactly one provider call: the compacted request. The 82,000-token
        # request never went to the wall.
        assert agent.client.chat.completions.create.call_count == 1
        # No exhaustion / defer — the pass succeeded within budget.
        assert not result.get("compression_exhausted")
        assert not result.get("compression_deferred")

    def test_fit_request_below_usable_budget_sends_without_compression(self, agent):
        """Estimate 79,000: below the 85,500 threshold AND below the 80,000
        usable budget. The guard must NOT fire — the request goes straight to
        the provider with no pre-API pass.

        (Regression: the guard is a second trigger on the same gate, not a
        blanket pre-send check that fires whenever a window is known.)
        """
        agent.context_compressor = _wall_compressor(150_000, 85_500)
        agent.max_tokens = 70_000
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="direct answer"),
        ]

        e1, e2, e3, e4 = _patch_estimates(est=79_000)
        with e1, e2, e3, e4, patch.object(agent, "_compress_context") as mock_compress, \
             patch.object(agent, "_persist_session"), \
             patch.object(agent, "_save_trajectory"), \
             patch.object(agent, "_cleanup_task_resources"):
            result = agent.run_conversation(
                "hello", conversation_history=list(_PREFILL)
            )

        mock_compress.assert_not_called()
        assert result.get("completed") is True
        assert result["final_response"] == "direct answer"

    def test_unknown_window_stays_fail_open(self, agent):
        """context_length = 0 (unknown window): the guard cannot judge fit,
        so it stays quiet even with a large estimate below the threshold —
        the legacy threshold trigger alone governs. Estimate 90,000 <
        threshold 100,000: no compression, straight to the provider."""
        agent.context_compressor = SimpleNamespace(
            protect_first_n=3,
            protect_last_n=20,
            threshold_tokens=100_000,
            context_length=0,  # unknown window
            last_prompt_tokens=0,
            should_compress=lambda t: t >= 100_000,
            should_defer_preflight_to_real_usage=lambda _t: False,
            get_active_compression_failure_cooldown=lambda: None,
        )
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="sent with unknown window"),
        ]

        e1, e2, e3, e4 = _patch_estimates(est=90_000)
        with e1, e2, e3, e4, patch.object(agent, "_compress_context") as mock_compress, \
             patch.object(agent, "_persist_session"), \
             patch.object(agent, "_save_trajectory"), \
             patch.object(agent, "_cleanup_task_resources"):
            result = agent.run_conversation(
                "hello", conversation_history=list(_PREFILL)
            )

        mock_compress.assert_not_called()
        assert result.get("completed") is True

    def test_guard_reuses_defer_predicate(self, agent):
        """The guard must stand down behind the same defer predicate the
        threshold trigger uses (#36718): if a recent real provider prompt
        proves the rough estimate is over-counting, neither trigger fires.

        Estimate 82,000 is in the guard's zone (usable 80,000 < 82,000 <
        threshold 85,500), but should_defer_preflight_to_real_usage says the
        last real prompt fit at a far lower count — so NO pre-send pass.
        """
        comp = _wall_compressor(150_000, 85_500)
        comp.should_defer_preflight_to_real_usage = lambda t: t > 60_000
        agent.context_compressor = comp
        agent.max_tokens = 70_000
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="deferred request went through"),
        ]

        e1, e2, e3, e4 = _patch_estimates(est=82_000)
        with e1, e2, e3, e4, patch.object(agent, "_compress_context") as mock_compress, \
             patch.object(agent, "_persist_session"), \
             patch.object(agent, "_save_trajectory"), \
             patch.object(agent, "_cleanup_task_resources"):
            result = agent.run_conversation(
                "hello", conversation_history=list(_PREFILL)
            )

        mock_compress.assert_not_called()
        assert result.get("completed") is True

    def test_over_threshold_still_triggers_unchanged(self, agent):
        """Estimate 90,000 > threshold 85,500 (and > usable 80,000): the
        threshold trigger fires as before — the guard's addition must not
        change the legacy trigger's behavior. One pass, then send."""
        agent.context_compressor = _wall_compressor(150_000, 85_500)
        agent.max_tokens = 70_000
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="threshold path"),
        ]

        est_values = iter([90_000, 30_000, 30_000])

        def _est(*_a, **_k):
            return next(est_values, 30_000)

        def _compress(messages, _system_message, **_kwargs):
            return (
                [{"role": "user", "content": "compact summary"}],
                "You are helpful.",
            )

        with (
            patch("agent.turn_context.estimate_request_tokens_rough", return_value=30_000),
            patch("agent.turn_context.estimate_messages_tokens_rough", return_value=30_000),
            patch("agent.conversation_loop.estimate_request_tokens_rough", side_effect=_est),
            patch("agent.conversation_loop.estimate_messages_tokens_rough", side_effect=_est),
            patch.object(agent, "_compress_context", side_effect=_compress) as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                "hello", conversation_history=list(_PREFILL)
            )

        mock_compress.assert_called_once()
        assert result.get("completed") is True
        assert agent.client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# Calibrated basis: the (rough, real) note carries the scale
# ---------------------------------------------------------------------------


class TestCalibratedRoughNote:
    def test_loop_rough_note_is_calibrated(self, agent):
        """The request-rough estimate stashed for the (rough, real) defer
        projection (``note_request_rough_estimate``) must be built on the
        calibrated basis — the same basis the prologue preflight and the fit
        guard use. A plain base estimate here would make the projection
        compare calibrated roughs against an uncalibrated anchor.

        Model "test-model" has token_estimate_scale 2.0; base estimate of
        the request is 50,000 → the note must be 100,000 (tools: none).
        """
        agent.context_compressor = SimpleNamespace(
            protect_first_n=3,
            protect_last_n=20,
            threshold_tokens=200_000,  # far above: no compression this pass
            context_length=1_000_000,  # fits: guard quiet
            last_prompt_tokens=0,
            should_compress=lambda t: t >= 200_000,
            should_defer_preflight_to_real_usage=lambda _t: False,
            get_active_compression_failure_cooldown=lambda: None,
            note_request_rough_estimate=MagicMock(),
        )
        agent.model = "test-model"
        agent.compression_token_estimate_scale = {
            "scales": {"test-model": 2.0},
        }
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="ok"),
        ]

        e1, e2, e3, e4 = _patch_estimates(est=50_000)
        with e1, e2, e3, e4, \
             patch.object(agent, "_persist_session"), \
             patch.object(agent, "_save_trajectory"), \
             patch.object(agent, "_cleanup_task_resources"):
            result = agent.run_conversation(
                "hello", conversation_history=list(_PREFILL)
            )

        assert result.get("completed") is True
        note = agent.context_compressor.note_request_rough_estimate
        assert note.call_count >= 1
        noted = [c.args[0] for c in note.call_args_list]
        # The loop's pre-API note is the calibrated figure: 50,000 * 2.0.
        assert any(v == 100_000 for v in noted), (
            f"expected a calibrated 100,000 rough note, got {noted}"
        )
