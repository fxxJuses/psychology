"""
RAG 评估模块

提供检索指标计算、生成质量评估、测试用例管理和结果报告。
"""

from .metrics import (
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
    hit_rate,
    average_precision,
    compute_all_retrieval_metrics,
)
from .test_cases import TestCase, TestSuite
from .generation_eval import (
    evaluate_faithfulness,
    evaluate_answer_relevance,
    GenerationEvaluator,
)
from .runner import RAGEvaluator, EvalResult
from .reporter import EvalReporter

__all__ = [
    # 检索指标
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "hit_rate",
    "average_precision",
    "compute_all_retrieval_metrics",
    # 测试用例
    "TestCase",
    "TestSuite",
    # 生成评估
    "evaluate_faithfulness",
    "evaluate_answer_relevance",
    "GenerationEvaluator",
    # 评估器 & 报告
    "RAGEvaluator",
    "EvalResult",
    "EvalReporter",
]
