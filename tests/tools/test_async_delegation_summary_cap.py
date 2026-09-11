"""Structural belt-and-suspenders: async_delegation's own finalize path caps
subagent summaries even when it has no live parent_agent handle to compute a
dynamic budget from (e.g. cronjob_tools' manual-run reporting, which dispatches
through dispatch_async_delegation directly rather than delegate_task's batch
path). Without this, a caller that forgets to self-truncate its own summary
text has nothing stopping an oversized, uncompactable message from entering a
session's history — see the delegate_tool-side cap this mirrors in
test_delegate_summary_budget.py (PR #9126).
"""

import os
import tempfile

from tools import async_delegation as ad
import tools.delegate_tool as dt


def test_cap_result_summary_trims_oversized_text_with_no_parent(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("HERMES_HOME", os.path.join(td, ".hermes"))
        big = "HEAD_MARKER\n" + ("X" * (dt.DEFAULT_MAX_SUMMARY_CHARS * 2)) + "\nTAIL_MARKER"
        result = {"task_index": 0, "summary": big, "status": "completed"}

        ad._cap_result_summary(result)

        assert result["summary_truncated"] is True
        assert len(result["summary"]) < len(big)
        assert "HEAD_MARKER" in result["summary"]
        assert "TAIL_MARKER" in result["summary"]
        path = result.get("summary_full_path")
        assert path and os.path.exists(path)
        with open(path, encoding="utf-8") as fh:
            assert fh.read() == big


def test_cap_result_summary_leaves_small_text_untouched():
    result = {"task_index": 0, "summary": "short and fine", "status": "completed"}
    ad._cap_result_summary(result)
    assert result["summary"] == "short and fine"
    assert "summary_truncated" not in result


def test_cap_batch_summaries_trims_each_oversized_entry(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("HERMES_HOME", os.path.join(td, ".hermes"))
        big = "HEAD_MARKER\n" + ("X" * (dt.DEFAULT_MAX_SUMMARY_CHARS * 2)) + "\nTAIL_MARKER"
        combined = {
            "results": [
                {"task_index": 0, "summary": big, "status": "completed"},
                {"task_index": 1, "summary": "small one", "status": "completed"},
            ]
        }

        ad._cap_batch_summaries(combined)

        assert combined["results"][0]["summary_truncated"] is True
        assert len(combined["results"][0]["summary"]) < len(big)
        assert "summary_truncated" not in combined["results"][1]


def test_cap_batch_summaries_tolerates_missing_results_list():
    combined = {"error": "no results at all"}
    ad._cap_batch_summaries(combined)  # must not raise
    assert combined == {"error": "no results at all"}
