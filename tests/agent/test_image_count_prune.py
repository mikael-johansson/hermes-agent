"""Count-based per-prompt image prune on the send path.

Native-vision sessions accumulate image content parts (one per
``vision_analyze`` fast-path result plus user-attached images).  Existing
aging is token-based only, so a provider with a *count* cap on per-prompt
images (e.g. a vLLM backend at 50) 400s once the outgoing request exceeds
the cap even at low token counts.  ``agent.max_images_per_prompt``
(default 64; 0/negative = disabled) bounds that: the newest N image parts
are always kept, older ones become short text stubs (+ on-disk path ref
when available).  The prune mutates the per-call ``api_messages`` copy
only — stored history is never touched.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

from agent.conversation_loop import _prune_image_parts_in_api_messages
from agent.image_routing import (
    DEFAULT_MAX_IMAGES_PER_PROMPT,
    max_images_per_prompt,
)


def _multimodal_envelope(path: str, summary: str = "native vision summary") -> Dict[str, Any]:
    """Shape of tools/vision_tools._build_native_vision_tool_result."""
    return {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": "Image loaded into your context."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
        "text_summary": summary,
        "meta": {"image_url": path, "size_bytes": 1234, "native_vision": True},
    }


def _image_part(url: str) -> Dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


def _build_history() -> List[Dict[str, Any]]:
    """Three oldest→newest multimodal messages:

    m0 — user turn, text + data-URL image parts (list content)
    m1 — tool turn, ``_multimodal`` envelope with a meta path ref
    m2 — user turn, a single data-URL image part (newest)
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe these"},
                _image_part("data:image/png;base64,OLD0"),
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": _multimodal_envelope("/tmp/img1.png", "s1"),
        },
        {
            "role": "user",
            "content": [_image_part("data:image/png;base64,NEW")],
        },
    ]


def _run(
    monkeypatch,
    api_messages: List[Dict[str, Any]],
    config: Dict[str, Any],
    logger=None,
):
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: config)
    return _prune_image_parts_in_api_messages(api_messages, logger=logger)


class TestPruneBehavior:
    def test_keeps_newest_prunes_oldest(self, monkeypatch):
        history = _build_history()
        api = copy.deepcopy(history)
        api[0]["api_content"] = "stale-bytes-for-m0"
        out = _run(monkeypatch, api, {"agent": {"max_images_per_prompt": 2}})

        # Nothing deleted: message count and roles preserved (tool_call_id
        # pairing must survive).
        assert len(out) == 3
        assert [m["role"] for m in out] == ["user", "tool", "user"]

        # Oldest image part (m0's data URL) → short text stub; the text
        # part in the same message survives.
        m0_content = out[0]["content"]
        assert isinstance(m0_content, list)
        assert m0_content[0] == {"type": "text", "text": "describe these"}
        assert m0_content[1]["type"] == "text"
        assert "image" in m0_content[1]["text"]
        assert "budget" in m0_content[1]["text"]
        assert "data:image/png;base64,OLD0" not in str(m0_content)
        # The rewrite dropped the stale api_content sidecar so replay can't
        # resend the pre-rewrite bytes.
        assert "api_content" not in out[0]

        # The two newest images (m1 envelope, m2 part) are intact,
        # byte-identical to the stored originals.
        assert out[1]["content"] == history[1]["content"]
        assert out[2]["content"] == history[2]["content"]

    def test_envelope_stub_carries_path_ref_and_summary(self, monkeypatch):
        history = _build_history()
        api = copy.deepcopy(history)
        out = _run(monkeypatch, api, {"agent": {"max_images_per_prompt": 1}})

        # Only the newest image (m2) survives.
        assert out[2]["content"] == history[2]["content"]

        # The pruned _multimodal envelope is replaced as a WHOLE by a
        # one-line stub: budget marker + text_summary prefix + on-disk ref.
        m1 = out[1]["content"]
        assert isinstance(m1, str)
        assert m1.startswith("[image removed to stay within the per-prompt image budget]")
        assert "s1" in m1
        assert "/tmp/img1.png" in m1
        # And the envelope's api_content sidecar was dropped.
        assert "api_content" not in out[1]

    def test_noop_at_or_below_cap(self, monkeypatch):
        history = _build_history()
        api = copy.deepcopy(history)
        logger = MagicMock()
        for cap in (3, 100):
            out = _run(monkeypatch, api, {"agent": {"max_images_per_prompt": cap}}, logger=logger)
            assert out == history, f"cap={cap} must not mutate"
        logger.info.assert_not_called()

    def test_disabled_when_zero_or_negative(self, monkeypatch):
        history = _build_history()
        api = copy.deepcopy(history)
        for cap in (0, -5):
            out = _run(monkeypatch, api, {"agent": {"max_images_per_prompt": cap}})
            assert out == history, f"cap={cap} must be a no-op"

    def test_stored_history_never_mutated(self, monkeypatch):
        history = _build_history()
        original = copy.deepcopy(history)
        api = copy.deepcopy(history)
        _run(monkeypatch, api, {"agent": {"max_images_per_prompt": 1}})
        assert history == original, "stored history must be untouched"

    def test_logs_one_line_per_firing(self, monkeypatch):
        history = _build_history()
        api = copy.deepcopy(history)
        logger = MagicMock()
        _run(monkeypatch, api, {"agent": {"max_images_per_prompt": 1}}, logger=logger)
        assert logger.info.call_count == 1
        # pruned=2, kept=1, cap=1
        args = str(logger.info.call_args)
        assert "2" in args and "1" in args

    def test_missing_and_broken_config_falls_back_to_default(self, monkeypatch):
        history = _build_history()
        # No agent section at all → default 64 → no-op for 3 images.
        api = copy.deepcopy(history)
        assert _run(monkeypatch, api, {}) == history
        # Non-dict junk in the agent section → default 64 → no-op.
        api = copy.deepcopy(history)
        assert _run(monkeypatch, api, {"agent": "nonsense"}) == history


class TestMaxImagesPerPromptConfig:
    def test_default_is_64(self):
        assert DEFAULT_MAX_IMAGES_PER_PROMPT == 64
        assert max_images_per_prompt({}) == 64
        assert max_images_per_prompt({"agent": {}}) == 64
        assert max_images_per_prompt(None) == 64

    def test_explicit_value_honored(self):
        assert max_images_per_prompt({"agent": {"max_images_per_prompt": 50}}) == 50

    def test_zero_and_negative_mean_disabled_not_default(self):
        assert max_images_per_prompt({"agent": {"max_images_per_prompt": 0}}) == 0
        assert max_images_per_prompt({"agent": {"max_images_per_prompt": -3}}) == 0

    def test_boolean_rejected_not_coerced(self):
        # bool subclasses int: YAML `max_images_per_prompt: true` must fall
        # back to the default, never coerce to 1.
        assert max_images_per_prompt({"agent": {"max_images_per_prompt": True}}) == 64
        assert max_images_per_prompt({"agent": {"max_images_per_prompt": False}}) == 64

    def test_fractional_float_rejected(self):
        assert (
            max_images_per_prompt({"agent": {"max_images_per_prompt": 50.5}}) == 64
        )

    def test_integral_float_and_numeric_string_accepted(self):
        assert max_images_per_prompt({"agent": {"max_images_per_prompt": 50.0}}) == 50
        assert max_images_per_prompt({"agent": {"max_images_per_prompt": "50"}}) == 50

    def test_garbage_falls_back_to_default(self):
        assert max_images_per_prompt({"agent": {"max_images_per_prompt": "lots"}}) == 64
        assert max_images_per_prompt({"agent": {"max_images_per_prompt": None}}) == 64


class TestConfigDefaults:
    def test_key_present_in_default_config(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["agent"]["max_images_per_prompt"] == 64


class TestWiring:
    def test_prune_runs_between_drop_thinking_and_cache_plan(self):
        """The prune must run on the per-call copy AFTER
        ``_drop_thinking_only_and_merge_users`` and BEFORE
        ``build_prompt_cache_plan`` so cache planning sees the final
        message shape (and the stored history is never rewritten)."""
        import agent.conversation_loop as cl

        src = Path(cl.__file__).read_text()
        i_drop_call = src.index("agent._drop_thinking_only_and_merge_users(")
        i_prune_call = src.index("api_messages = _prune_image_parts_in_api_messages(")
        # There are two build_prompt_cache_plan( call sites (a helper + the
        # run_conversation send path). Take the one that follows the prune —
        # that is the cache plan the pruned request feeds into.
        i_cache_plan = src.index("build_prompt_cache_plan(", i_prune_call)
        assert i_drop_call < i_prune_call < i_cache_plan
