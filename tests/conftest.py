"""
测试公共 fixtures。
"""

import pytest
from rag_agent.evaluation.test_cases import TestSuite, TestCase


@pytest.fixture
def default_suite() -> TestSuite:
    """加载默认测试用例集"""
    return TestSuite.load_default()


@pytest.fixture
def empty_suite() -> TestSuite:
    """空测试用例集"""
    return TestSuite(name="empty")


@pytest.fixture
def sample_case() -> TestCase:
    """单条示例用例"""
    return TestCase(
        id="tc_test_001",
        category="factual_lookup",
        question="测试问题？",
        reference_answer="测试答案。",
        expected_keywords=["测试"],
        relevant_sources=["测试书"],
        description="这是一条测试用例",
    )


# ---------- 检索指标测试数据 ----------

@pytest.fixture
def relevant_ids():
    """相关文档 ID 集合 (3个)"""
    return {"a", "b", "c"}


@pytest.fixture
def ranked_ids():
    """检索结果排名 (a, x, b, y, c, z)"""
    return ["a", "x", "b", "y", "c", "z"]


@pytest.fixture
def no_match_ids():
    """全不相关的检索结果"""
    return ["x", "y", "z"]


@pytest.fixture
def graded_relevance():
    """分级相关性评分"""
    return {"a": 3.0, "b": 2.0, "c": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}


@pytest.fixture
def empty_relevant():
    """空相关文档集合"""
    return set()


@pytest.fixture
def empty_ranked():
    """空检索结果"""
    return []
