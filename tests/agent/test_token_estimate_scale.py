"""Per-model token-estimate calibration (``compression.token_estimate_scale``).

The base rough estimator assumes ~4 chars/token (English-centric). Dense
workloads — JSON dumps, tracebacks, markdown tables — tokenize far denser
(measured ~2.5 chars/token on a Qwen BPE), so the estimate *undercounts* the
real prompt by ~1.5-2.5x. That is the dangerous direction for a provider with
a hard context wall: the pre-flight trigger and the pre-send fit guard both run
on the estimate, so an undercount lets a session reach the wall before
compaction fires.

``token_estimate_scale`` multiplies the base estimate at the *aggregate* level
so it tracks the real tokenizer. ``1.0`` (the default) must be byte-identical
to the historical behavior — these tests pin that contract plus the new
calibration and the pre-send fit predicate.
"""

import math

from agent.model_metadata import (
    apply_token_estimate_scale,
    estimate_messages_tokens_rough,
    estimate_request_tokens_rough,
    prompt_fits_window,
    resolve_token_estimate_scale,
)


# =========================================================================
# apply_token_estimate_scale
# =========================================================================


class TestApplyTokenEstimateScale:
    def test_zero_tokens_is_zero_for_any_scale(self):
        assert apply_token_estimate_scale(0, 1.6) == 0
        assert apply_token_estimate_scale(0, 3.0) == 0

    def test_identity_scale_is_byte_identical(self):
        # The default (no calibration) must NOT move the estimate.
        for n in (0, 1, 3, 4, 100, 112512, 200000):
            assert apply_token_estimate_scale(n, 1.0) == n
            assert apply_token_estimate_scale(n, None) == n

    def test_scale_up_ceil(self):
        # 100 * 1.6 = 160 exactly -> 160
        assert apply_token_estimate_scale(100, 1.6) == 160
        # 10 * 1.1 = 11.0 (float 11.000...0) -> 11
        assert apply_token_estimate_scale(10, 1.1) == 11
        # 3 * 1.5 = 4.5 -> ceil 5 (the x % 1 == 0.0-on-non-integer edge the
        # old `int(x) + (1 if x % 1 else 0)` mishandled).
        assert apply_token_estimate_scale(3, 1.5) == 5
        # 1 * 2.5 -> 3
        assert apply_token_estimate_scale(1, 2.5) == 3
        # 112512 * 1.6 = 180019.2 -> 180020
        assert apply_token_estimate_scale(112512, 1.6) == 180020

    def test_never_floors_below_scaled_value(self):
        # A calibration must round the scaled value UP, never down.
        assert apply_token_estimate_scale(1, 1.2) == 2  # 1.2 -> 2
        assert apply_token_estimate_scale(7, 1.3) == 10  # 9.1 -> 10

    def test_invalid_scale_degrades_to_identity(self):
        # Genuinely invalid scales degrade to identity (no scaling):
        #   * values float() cannot coerce (TypeError/ValueError), and
        #   * non-positive values.
        assert apply_token_estimate_scale(50, "x") == 50  # non-numeric str
        assert apply_token_estimate_scale(50, [1.6]) == 50  # list -> TypeError
        assert apply_token_estimate_scale(50, object()) == 50  # uncoercible
        assert apply_token_estimate_scale(50, -1) == 50  # non-positive
        assert apply_token_estimate_scale(50, 0) == 50  # non-positive
        # A numeric *string* is a legitimate calibration a user may write
        # quoted (``token_estimate_scale: "1.6"``). It is coerced like a
        # float so the calibration is NOT silently dropped — dropping it
        # would undercount and defeat the pre-send fit guard, so it scales
        # (80) rather than degrading to identity.
        assert apply_token_estimate_scale(50, "1.6") == 80


# =========================================================================
# resolve_token_estimate_scale
# =========================================================================


class TestResolveTokenEstimateScale:
    def test_none_config_defaults_to_one(self):
        assert resolve_token_estimate_scale("qwen", None) == 1.0

    def test_scalar_config(self):
        assert resolve_token_estimate_scale("qwen", 1.6) == 1.6
        assert resolve_token_estimate_scale("qwen", {"scale": 1.8}) == 1.8

    def test_per_model_override_wins(self):
        cfg = {"scale": 1.2, "scales": {"qwen3-27b": 1.7, "gpt-4o": 1.1}}
        assert resolve_token_estimate_scale("qwen3-27b", cfg) == 1.7
        assert resolve_token_estimate_scale("gpt-4o", cfg) == 1.1

    def test_case_insensitive_per_model_match(self):
        cfg = {"scales": {"Qwen3.8-27B": 1.7}}
        # user typed the model in a different case than the key
        assert resolve_token_estimate_scale("qwen3.8-27b", cfg) == 1.7

    def test_no_match_falls_back_to_global(self):
        cfg = {"scale": 1.2, "scales": {"qwen3-27b": 1.7}}
        assert resolve_token_estimate_scale("gpt-4o", cfg) == 1.2

    def test_none_model_uses_global(self):
        cfg = {"scale": 1.2, "scales": {"qwen3-27b": 1.7}}
        assert resolve_token_estimate_scale(None, cfg) == 1.2

    def test_invalid_values_degrade_to_one(self):
        assert resolve_token_estimate_scale("m", {"scale": "x"}) == 1.0
        assert resolve_token_estimate_scale("m", {"scale": -5}) == 1.0
        assert resolve_token_estimate_scale("m", {"scales": {"m": "bad"}}) == 1.0
        assert resolve_token_estimate_scale("m", {"scales": {"m": 0}}) == 1.0


# =========================================================================
# prompt_fits_window
# =========================================================================


class TestPromptFitsWindow:
    def test_unknown_window_fits_fail_open(self):
        # context_length 0 / negative => cannot prove overflow => (True, 0)
        assert prompt_fits_window(5_000_000, 0) == (True, 0)
        assert prompt_fits_window(5_000_000, -1) == (True, 0)
        assert prompt_fits_window(5_000_000, None) == (True, 0)

    def test_no_output_cap_reserves_min_response(self):
        # No cap: usable = window - min_response (default 512).
        fits, usable = prompt_fits_window(10_000, 20_000)
        assert fits is True
        assert usable == 20_000 - 512

    def test_no_output_cap_overflows(self):
        fits, usable = prompt_fits_window(20_000, 20_000)
        assert fits is False
        assert usable == 20_000 - 512

    def test_output_cap_reserved_from_window(self):
        # Cap present: the provider already reserves it, so the prompt must fit
        # in window - cap.
        fits, usable = prompt_fits_window(140_000, 150_000, max_output_tokens=65_536)
        assert usable == 150_000 - 65_536  # 84464
        assert fits is False  # 140k > 84k

        fits, usable = prompt_fits_window(80_000, 150_000, max_output_tokens=65_536)
        assert fits is True
        assert usable == 84_464

    def test_output_cap_zero_uses_min_response(self):
        # max_output_tokens 0 / None both fall back to the min-response floor.
        assert prompt_fits_window(1_000, 5_000, max_output_tokens=0)[1] == 5_000 - 512
        assert prompt_fits_window(1_000, 5_000, max_output_tokens=None)[1] == 5_000 - 512

    def test_safety_margin(self):
        _, usable = prompt_fits_window(
            1_000, 5_000, max_output_tokens=100, safety_margin=50
        )
        assert usable == 5_000 - 100 - 50

    def test_exactly_fits_is_fits(self):
        # Boundary: prompt == usable input budget => fits (<=, not <).
        fits, usable = prompt_fits_window(
            150_000 - 512, 150_000
        )
        assert fits is True
        assert usable == 150_000 - 512
        fits, _ = prompt_fits_window(150_000 - 511, 150_000)
        assert fits is False


# =========================================================================
# scale plumbed through the estimators
# =========================================================================


class TestEstimatorScalePlumbing:
    def test_messages_scale_scales_the_total(self):
        # A dense single-message history. The exact base depends on the
        # message-shadow serialization (role + content wrapper, not just the
        # content), so pin the *scaling contract* — ceil(base * scale) —
        # rather than an absolute token count.
        msgs = [{"role": "user", "content": "x" * 4000}]
        base = estimate_messages_tokens_rough(msgs)
        assert base > 0
        assert estimate_messages_tokens_rough(msgs, scale=1.6) == math.ceil(base * 1.6)
        assert estimate_messages_tokens_rough(msgs, scale=2.5) == math.ceil(base * 2.5)

    def test_messages_scale_default_is_identity(self):
        # The default (no calibration) must be byte-identical to the unscaled
        # estimate — this is the "1.0 preserves historical behavior" contract.
        msgs = [{"role": "user", "content": "x" * 4000}]
        base = estimate_messages_tokens_rough(msgs)
        assert base == estimate_messages_tokens_rough(msgs, scale=1.0)
        assert base == estimate_messages_tokens_rough(msgs, scale=None)

    def test_request_scale_covers_system_messages_and_tools(self):
        msgs = [{"role": "user", "content": "x" * 4000}]  # 1000
        system = "s" * 400  # 100
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "t",
                    "description": "d",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        base = estimate_request_tokens_rough(msgs, system_prompt=system, tools=tools)
        scaled = estimate_request_tokens_rough(
            msgs, system_prompt=system, tools=tools, scale=2.0
        )
        # scale applies to the aggregate total (system + messages + tools).
        assert scaled == base * 2
        # default is identity
        assert estimate_request_tokens_rough(
            msgs, system_prompt=system, tools=tools
        ) == base
