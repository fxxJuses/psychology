"""
test_cases.py 测试用例集测试

覆盖：
- TestSuite 加载、遍历、过滤
- TestCase 数据完整性
- 各类别用例数量
- 负样本独立性
"""

import pytest
from rag_agent.evaluation.test_cases import (
    TestCase,
    TestSuite,
    CATEGORY_FACTUAL,
    CATEGORY_DEFINITION,
    CATEGORY_COMPARISON,
    CATEGORY_MULTI_HOP,
    CATEGORY_SUMMARY,
    CATEGORY_NEGATIVE,
)


class TestTestCase:
    def test_create_minimal(self):
        """创建最小测试用例"""
        tc = TestCase(
            id="tc_001",
            category=CATEGORY_FACTUAL,
            question="测试问题？",
            reference_answer="测试答案。",
        )
        assert tc.id == "tc_001"
        assert tc.category == CATEGORY_FACTUAL
        assert tc.question == "测试问题？"
        assert tc.expected_keywords == []
        assert tc.relevant_sources == []
        assert tc.description == ""

    def test_create_full(self, sample_case):
        """创建完整测试用例"""
        assert sample_case.id == "tc_test_001"
        assert sample_case.category == CATEGORY_FACTUAL
        assert len(sample_case.expected_keywords) == 1
        assert len(sample_case.relevant_sources) == 1
        assert "测试" in sample_case.description

    def test_reference_answer_present(self):
        """参考答案不应为空（负样本除外）"""
        tc = TestCase(
            id="tc_x",
            category=CATEGORY_FACTUAL,
            question="问题",
            reference_answer="有答案",
        )
        assert len(tc.reference_answer) > 0


class TestTestSuite:
    def test_empty_suite(self, empty_suite):
        """空测试集"""
        assert len(empty_suite) == 0
        assert empty_suite.name == "empty"
        assert empty_suite.categories == []

    def test_add_case(self, empty_suite, sample_case):
        """添加用例"""
        empty_suite.add_case(sample_case)
        assert len(empty_suite) == 1
        assert empty_suite.test_cases[0] is sample_case

    def test_iteration(self, empty_suite, sample_case):
        """迭代测试"""
        empty_suite.add_case(sample_case)
        cases = list(empty_suite)
        assert len(cases) == 1
        assert cases[0].id == "tc_test_001"

    def test_get_by_category(self, empty_suite, sample_case):
        """按类别筛选"""
        empty_suite.add_case(sample_case)
        empty_suite.add_case(TestCase(
            id="tc_002", category=CATEGORY_DEFINITION,
            question="什么？", reference_answer="答案。",
        ))
        factual = empty_suite.get_by_category(CATEGORY_FACTUAL)
        assert len(factual) == 1
        assert factual[0].id == "tc_test_001"

        definition = empty_suite.get_by_category(CATEGORY_DEFINITION)
        assert len(definition) == 1

        # 不存在的类别
        empty = empty_suite.get_by_category("nonexistent")
        assert len(empty) == 0


class TestDefaultSuite:
    """测试默认测试集"""

    def test_load_succeeds(self, default_suite):
        """默认测试集加载成功"""
        suite = default_suite
        assert suite.name == "default"
        assert len(suite) >= 12  # 至少 12 条

    def test_all_categories_present(self, default_suite):
        """所有 6 个类别都存在"""
        expected_cats = {
            CATEGORY_FACTUAL, CATEGORY_DEFINITION, CATEGORY_COMPARISON,
            CATEGORY_MULTI_HOP, CATEGORY_SUMMARY, CATEGORY_NEGATIVE,
        }
        actual_cats = set(default_suite.categories)
        assert actual_cats == expected_cats

    def test_category_counts(self, default_suite):
        """各类别数量符合预期"""
        assert len(default_suite.get_by_category(CATEGORY_FACTUAL)) == 3
        assert len(default_suite.get_by_category(CATEGORY_DEFINITION)) == 3
        assert len(default_suite.get_by_category(CATEGORY_COMPARISON)) == 2
        assert len(default_suite.get_by_category(CATEGORY_MULTI_HOP)) == 2
        assert len(default_suite.get_by_category(CATEGORY_SUMMARY)) == 2
        assert len(default_suite.get_by_category(CATEGORY_NEGATIVE)) == 3

    def test_unique_ids(self, default_suite):
        """所有用例 ID 唯一"""
        ids = [tc.id for tc in default_suite]
        assert len(ids) == len(set(ids)), f"重复 ID: {ids}"

    def test_non_empty_questions(self, default_suite):
        """所有问题非空"""
        for tc in default_suite:
            assert len(tc.question) > 0, f"{tc.id}: 问题为空"

    def test_reference_answers(self, default_suite):
        """非负样本用例应有参考答案"""
        for tc in default_suite:
            if tc.category != CATEGORY_NEGATIVE:
                assert len(tc.reference_answer) > 0, f"{tc.id}: 参考答案为空"

    def test_negative_cases_no_sources(self, default_suite):
        """负样本不应指定 relevant_sources"""
        for tc in default_suite.get_by_category(CATEGORY_NEGATIVE):
            assert tc.relevant_sources == [], f"{tc.id}: 负样本不应有 relevant_sources"

    def test_factual_cases_have_keywords(self, default_suite):
        """事实检索用例应有期望关键词"""
        for tc in default_suite.get_by_category(CATEGORY_FACTUAL):
            assert len(tc.expected_keywords) > 0, f"{tc.id}: missing keywords"

    def test_descriptions_present(self, default_suite):
        """每条用例有说明"""
        for tc in default_suite:
            assert len(tc.description) > 0, f"{tc.id}: 缺少说明"


class TestSuiteFiltering:
    """测试 TestSuite 的高级操作"""

    def test_load_default_is_idempotent(self, default_suite):
        """重复加载结果一致"""
        suite1 = default_suite
        suite2 = TestSuite.load_default()
        assert len(suite1) == len(suite2)
        ids1 = [tc.id for tc in suite1]
        ids2 = [tc.id for tc in suite2]
        assert ids1 == ids2

    def test_get_by_category_returns_copy(self, default_suite):
        """get_by_category 不影响原测试集"""
        factual = default_suite.get_by_category(CATEGORY_FACTUAL)
        factual.append(TestCase(
            id="tc_fake", category=CATEGORY_FACTUAL,
            question="?", reference_answer="!",
        ))
        # 原测试集不应被修改
        assert len(default_suite.get_by_category(CATEGORY_FACTUAL)) == 3
