"""
reporter.py 报告器测试

覆盖：
- JSON 导出/导入
- 表格输出（验证不崩溃）
- EvalResult 空数据与完整数据
"""

import json
import os
import tempfile
import pytest
from rag_agent.evaluation.reporter import EvalReporter
from rag_agent.evaluation.runner import EvalResult


@pytest.fixture
def sample_retrieval_results():
    """模拟检索评估结果"""
    return {
        "vector": {
            "recall@1": 0.33, "precision@1": 0.67, "ndcg@1": 0.45, "hit@1": 0.67,
            "recall@3": 0.56, "precision@3": 0.44, "ndcg@3": 0.52, "hit@3": 0.89,
            "recall@5": 0.67, "precision@5": 0.33, "ndcg@5": 0.58, "hit@5": 0.94,
            "recall@8": 0.78, "precision@8": 0.22, "ndcg@8": 0.63, "hit@8": 1.00,
            "mrr": 0.75, "map": 0.62,
        },
        "bm25": {
            "recall@1": 0.22, "precision@1": 0.44, "ndcg@1": 0.30, "hit@1": 0.44,
            "recall@3": 0.44, "precision@3": 0.33, "ndcg@3": 0.41, "hit@3": 0.72,
            "recall@5": 0.56, "precision@5": 0.27, "ndcg@5": 0.48, "hit@5": 0.83,
            "recall@8": 0.67, "precision@8": 0.19, "ndcg@8": 0.53, "hit@8": 0.89,
            "mrr": 0.58, "map": 0.48,
        },
        "hybrid": {
            "recall@1": 0.44, "precision@1": 0.89, "ndcg@1": 0.60, "hit@1": 0.89,
            "recall@3": 0.72, "precision@3": 0.56, "ndcg@3": 0.67, "hit@3": 0.94,
            "recall@5": 0.83, "precision@5": 0.40, "ndcg@5": 0.72, "hit@5": 1.00,
            "recall@8": 0.94, "precision@8": 0.28, "ndcg@8": 0.77, "hit@8": 1.00,
            "mrr": 0.89, "map": 0.74,
        },
    }


@pytest.fixture
def sample_generation_results():
    """模拟生成评估结果"""
    return {
        "faithfulness": 0.78,
        "answer_relevance": 0.85,
    }


@pytest.fixture
def sample_per_category():
    """模拟按类别评估结果"""
    return {
        "factual_lookup": {
            "vector": {"recall@5": 0.75, "mrr": 0.80},
            "bm25": {"recall@5": 0.60, "mrr": 0.65},
            "hybrid": {"recall@5": 0.88, "mrr": 0.92},
        },
        "definition": {
            "vector": {"recall@5": 0.55, "mrr": 0.62},
            "bm25": {"recall@5": 0.45, "mrr": 0.48},
            "hybrid": {"recall@5": 0.70, "mrr": 0.75},
        },
    }


@pytest.fixture
def full_eval_result(sample_retrieval_results, sample_generation_results, sample_per_category):
    """完整评估结果"""
    return EvalResult(
        test_suite_name="test_suite",
        retrieval=sample_retrieval_results,
        generation=sample_generation_results,
        per_category=sample_per_category,
    )


@pytest.fixture
def empty_eval_result():
    """空评估结果"""
    return EvalResult(test_suite_name="empty")


class TestPrintRetrievalTable:
    def test_with_data_does_not_crash(self, sample_retrieval_results):
        """有数据时不崩溃"""
        EvalReporter.print_retrieval_table(sample_retrieval_results)

    def test_with_empty_dict(self):
        """空 dict 不崩溃"""
        EvalReporter.print_retrieval_table({})

    def test_single_mode(self):
        """单模式"""
        results = {"vector": {"recall@3": 0.5, "mrr": 0.6}}
        EvalReporter.print_retrieval_table(results)


class TestPrintGenerationTable:
    def test_with_data_does_not_crash(self, sample_generation_results):
        """有数据时不崩溃"""
        EvalReporter.print_generation_table(sample_generation_results)

    def test_with_empty_dict(self):
        """空 dict 不崩溃"""
        EvalReporter.print_generation_table({})

    def test_partial_results(self):
        """部分指标"""
        results = {"faithfulness": 0.5}
        EvalReporter.print_generation_table(results)


class TestPrintCategoryTable:
    def test_with_data_does_not_crash(self, sample_per_category):
        """有数据时不崩溃"""
        EvalReporter.print_category_table(sample_per_category, metric="recall@5")
        EvalReporter.print_category_table(sample_per_category, metric="mrr")

    def test_with_empty_dict(self):
        """空 dict 不崩溃"""
        EvalReporter.print_category_table({}, metric="recall@5")


class TestPrintFullReport:
    def test_full_report_does_not_crash(self, full_eval_result):
        """完整报告不崩溃"""
        EvalReporter.print_full_report(full_eval_result, show_per_category=True)

    def test_without_category(self, full_eval_result):
        """不显示类别拆分"""
        EvalReporter.print_full_report(full_eval_result, show_per_category=False)

    def test_empty_report_does_not_crash(self, empty_eval_result):
        """空报告不崩溃"""
        EvalReporter.print_full_report(empty_eval_result)

    def test_only_retrieval(self, sample_retrieval_results):
        """仅检索结果"""
        result = EvalResult(
            test_suite_name="test",
            retrieval=sample_retrieval_results,
        )
        EvalReporter.print_full_report(result)

    def test_only_generation(self, sample_generation_results):
        """仅生成结果"""
        result = EvalResult(
            test_suite_name="test",
            generation=sample_generation_results,
        )
        EvalReporter.print_full_report(result)


class TestExportJSON:
    def test_export_and_reload(self, full_eval_result):
        """导出 JSON 后能重新加载"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            tmp_path = f.name
        try:
            EvalReporter.export_json(full_eval_result, tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["test_suite"] == "test_suite"
            assert "retrieval" in data
            assert "generation" in data
            assert "per_category" in data
            # 验证检索数据完整性
            assert "vector" in data["retrieval"]
            assert "hybrid" in data["retrieval"]
            assert "recall@5" in data["retrieval"]["hybrid"]
            # 验证生成数据
            assert data["generation"]["faithfulness"] == pytest.approx(0.78)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_export_empty(self, empty_eval_result):
        """导出空的评估结果"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            tmp_path = f.name
        try:
            EvalReporter.export_json(empty_eval_result, tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["test_suite"] == "empty"
            assert data["retrieval"] == {}
            assert data["generation"] == {}
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_export_is_valid_json(self, full_eval_result):
        """导出的文件是有效 JSON"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            tmp_path = f.name
        try:
            EvalReporter.export_json(full_eval_result, tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as f:
                data = f.read()
            # 验证 JSON 解析不报错
            parsed = json.loads(data)
            assert isinstance(parsed, dict)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestEvalResult:
    def test_defaults(self):
        """EvalResult 默认值"""
        result = EvalResult(test_suite_name="test")
        assert result.test_suite_name == "test"
        assert result.retrieval == {}
        assert result.generation == {}
        assert result.case_results == []
        assert result.per_category == {}

    def test_field_assignment(self):
        """字段赋值"""
        result = EvalResult(test_suite_name="test")
        result.retrieval["vector"] = {"recall@1": 0.5}
        result.generation["faithfulness"] = 0.8
        assert result.retrieval["vector"]["recall@1"] == pytest.approx(0.5)
        assert result.generation["faithfulness"] == pytest.approx(0.8)
