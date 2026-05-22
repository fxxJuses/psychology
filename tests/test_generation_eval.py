"""
generation_eval.py 生成评估测试

覆盖：
- _parse_score_response 解析各种格式
- evaluate_faithfulness 空输入
- evaluate_answer_relevance 空输入
- GenerationEvaluator 正常流程
"""

import pytest
from unittest.mock import MagicMock, patch
from rag_agent.evaluation.generation_eval import (
    _parse_score_response,
    evaluate_faithfulness,
    evaluate_answer_relevance,
    GenerationEvaluator,
    FAITHFULNESS_SYSTEM,
    FAITHFULNESS_USER,
    RELEVANCE_SYSTEM,
    RELEVANCE_USER,
)


class TestParseScoreResponse:
    def test_standard_format(self):
        """标准格式：分数 + 理由"""
        text = "分数: 8\n理由: 回答基本忠实于上下文"
        result = _parse_score_response(text)
        assert result["score"] == pytest.approx(8.0)
        assert "忠实" in result["reasoning"]

    def test_chinese_colon(self):
        """中文冒号"""
        text = "分数：7\n理由：部分内容无法从上下文中确认"
        result = _parse_score_response(text)
        assert result["score"] == pytest.approx(7.0)
        assert len(result["reasoning"]) > 0

    def test_no_reasoning_line(self):
        """无"理由"行，取全文"""
        text = "分数: 9"
        result = _parse_score_response(text)
        assert result["score"] == pytest.approx(9.0)
        assert result["reasoning"] == "分数: 9"

    def test_score_first_only(self):
        """多行分数只取第一个"""
        text = "分数: 6\n分数: 10\n理由: 测试"
        result = _parse_score_response(text)
        assert result["score"] == pytest.approx(6.0)

    def test_no_score(self):
        """无法提取分数 → 0"""
        text = "这个回答很好"
        result = _parse_score_response(text)
        assert result["score"] == pytest.approx(0.0)

    def test_negative_score_clamped(self):
        """负数被 clamp 到 0"""
        text = "分数: -5\n理由: 完全不相关"
        result = _parse_score_response(text)
        assert result["score"] == pytest.approx(0.0)

    def test_large_score_clamped(self):
        """超过 10 的分数被 clamp"""
        text = "分数: 999\n理由: 完美"
        result = _parse_score_response(text)
        assert result["score"] == pytest.approx(10.0)

    def test_extra_text_around_score(self):
        """分数周围有额外文字"""
        text = "我认为分数: 7 分\n理由: 还行"
        result = _parse_score_response(text)
        assert result["score"] == pytest.approx(7.0)

    def test_reasoning_truncated(self):
        """超长理由被截断到 500 字符"""
        text = f"分数: 5\n理由: {'x' * 600}"
        result = _parse_score_response(text)
        assert len(result["reasoning"]) <= 500


class TestFaithfulness:
    def test_empty_answer(self):
        """空答案 → 0.0"""
        llm = MagicMock()
        result = evaluate_faithfulness("问题？", "", ["上下文"], llm)
        assert result["score"] == pytest.approx(0.0)
        assert len(result["reasoning"]) > 0

    def test_empty_contexts(self):
        """空上下文 → 0.0"""
        llm = MagicMock()
        result = evaluate_faithfulness("问题？", "答案", [], llm)
        assert result["score"] == pytest.approx(0.0)

    def test_both_empty(self):
        """答案和上下文都为空 → 0.0"""
        llm = MagicMock()
        result = evaluate_faithfulness("问题？", "", [], llm)
        assert result["score"] == pytest.approx(0.0)

    def test_llm_error_returns_zero(self):
        """LLM 调用失败 → 0.0"""
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("API error")
        result = evaluate_faithfulness("问题？", "答案", ["上下文"], llm)
        assert result["score"] == pytest.approx(0.0)
        assert "出错" in result["reasoning"]

    def test_normal_response(self):
        """正常 LLM 响应"""
        llm = MagicMock()
        llm.invoke.return_value.content = "分数: 8\n理由: 回答忠实于上下文"
        result = evaluate_faithfulness("问题？", "答案", ["上下文"], llm)
        assert result["score"] == pytest.approx(0.8)
        assert "忠实" in result["reasoning"]

    def test_context_truncation(self):
        """长上下文被截断"""
        llm = MagicMock()
        llm.invoke.return_value.content = "分数: 10\n理由: 完全一致"
        long_ctx = ["x" * 2000] * 10  # 10 个长上下文，但只取前 5 个，每个截断 800
        result = evaluate_faithfulness("问题？", "答案", long_ctx, llm)
        assert result["score"] == pytest.approx(1.0)


class TestAnswerRelevance:
    def test_empty_answer(self):
        """空答案 → 0.0"""
        llm = MagicMock()
        result = evaluate_answer_relevance("问题？", "", llm)
        assert result["score"] == pytest.approx(0.0)

    def test_llm_error_returns_zero(self):
        """LLM 调用失败 → 0.0"""
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("timeout")
        result = evaluate_answer_relevance("问题？", "答案", llm)
        assert result["score"] == pytest.approx(0.0)
        assert "出错" in result["reasoning"]

    def test_high_relevance(self):
        """高度相关的回答"""
        llm = MagicMock()
        llm.invoke.return_value.content = "分数: 9\n理由: 直接回应了问题"
        result = evaluate_answer_relevance("问题？", "完美答案", llm)
        assert result["score"] == pytest.approx(0.9)

    def test_low_relevance(self):
        """不相关的回答"""
        llm = MagicMock()
        llm.invoke.return_value.content = "分数: 2\n理由: 答非所问"
        result = evaluate_answer_relevance("问题？", "无关答案", llm)
        assert result["score"] == pytest.approx(0.2)


class TestGenerationEvaluator:
    def test_evaluate_returns_all_fields(self):
        """评估返回完整字段"""
        llm = MagicMock()
        # 模拟两次 invoke：第一次是 faithfulness，第二次是 relevance
        llm.invoke.side_effect = [
            MagicMock(content="分数: 7\n理由: 基本忠实"),
            MagicMock(content="分数: 8\n理由: 比较切题"),
        ]
        evaluator = GenerationEvaluator(llm)
        result = evaluator.evaluate("问题？", "答案", ["上下文"])
        assert "faithfulness" in result
        assert "faithfulness_reasoning" in result
        assert "answer_relevance" in result
        assert "relevance_reasoning" in result
        assert 0 <= result["faithfulness"] <= 1
        assert 0 <= result["answer_relevance"] <= 1

    def test_scores_in_range(self):
        """所有分数在 [0,1] 范围内"""
        llm = MagicMock()
        llm.invoke.side_effect = [
            MagicMock(content="分数: 10\n理由: 完美"),
            MagicMock(content="分数: 0\n理由: 完全不相关"),
        ]
        evaluator = GenerationEvaluator(llm)
        result = evaluator.evaluate("问题？", "答案", ["上下文"])
        assert result["faithfulness"] == pytest.approx(1.0)
        assert result["answer_relevance"] == pytest.approx(0.0)


class TestPrompts:
    """验证 Prompt 模板完整性"""

    def test_faithfulness_prompts_non_empty(self):
        """Faithfulness prompt 模板非空"""
        assert len(FAITHFULNESS_SYSTEM) > 0
        assert "{question}" in FAITHFULNESS_USER
        assert "{context}" in FAITHFULNESS_USER
        assert "{answer}" in FAITHFULNESS_USER

    def test_relevance_prompts_non_empty(self):
        """Relevance prompt 模板非空"""
        assert len(RELEVANCE_SYSTEM) > 0
        assert "{question}" in RELEVANCE_USER
        assert "{answer}" in RELEVANCE_USER
