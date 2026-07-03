"""
test_evaluator.py — tests for app/evaluator.py

Uses a fake runnable LLM (spec=Runnable) so the JSON-parsing, clamping, and
fail-open paths are exercised without calling a real model.
"""
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from evaluator import EvaluationResult, evaluate_answer


def _fake_llm(mocker, content=None, raise_exc=None):
    llm = mocker.MagicMock(spec=Runnable)
    if raise_exc is not None:
        llm.invoke.side_effect = raise_exc
    else:
        llm.invoke.return_value = AIMessage(content=content)
    return llm


# ── EvaluationResult ──────────────────────────────────────────────────────────

class TestEvaluationResult:
    def test_overall_is_mean_of_three(self):
        r = EvaluationResult(1.0, 0.5, 0.0)
        assert r.overall == pytest.approx(0.5, abs=1e-6)

    def test_as_dict_has_all_keys(self):
        r = EvaluationResult(0.9, 0.8, 0.7, reason="ok")
        d = r.as_dict()
        assert set(d) == {"relevance", "completeness", "groundedness", "overall", "reason"}


# ── evaluate_answer ───────────────────────────────────────────────────────────

class TestEvaluateAnswer:
    def test_parses_scores(self, mocker):
        llm = _fake_llm(
            mocker,
            content='{"relevance": 0.9, "completeness": 0.8, "groundedness": 1.0, "reason": "good"}',
        )
        r = evaluate_answer(llm, "q", "a", "ctx")
        assert (r.relevance, r.completeness, r.groundedness) == (0.9, 0.8, 1.0)
        assert r.failed is False

    def test_clamps_out_of_range_scores(self, mocker):
        llm = _fake_llm(
            mocker,
            content='{"relevance": 1.5, "completeness": -0.3, "groundedness": 0.5}',
        )
        r = evaluate_answer(llm, "q", "a", "ctx")
        assert r.relevance == 1.0
        assert r.completeness == 0.0
        assert r.groundedness == 0.5

    def test_fails_open_on_bad_json(self, mocker):
        llm = _fake_llm(mocker, content="not json at all")
        r = evaluate_answer(llm, "q", "a", "ctx")
        assert r.failed is True
        assert r.overall == 1.0

    def test_fails_open_on_exception(self, mocker):
        llm = _fake_llm(mocker, raise_exc=RuntimeError("boom"))
        r = evaluate_answer(llm, "q", "a", "ctx")
        assert r.failed is True
        assert (r.relevance, r.completeness, r.groundedness) == (1.0, 1.0, 1.0)
