"""
metrics.py 指标函数单元测试

覆盖：
- Recall@K, Precision@K: 正常 / 边界 / 空集
- MRR: 第一个命中 / 未命中 / 空集
- NDCG@K: 分级评分 / 理想排序
- Hit Rate: 命中 / 未命中
- MAP: 完整计算
- compute_all_retrieval_metrics: 批量计算
"""

import pytest
from rag_agent.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
    hit_rate,
    average_precision,
    compute_all_retrieval_metrics,
)


# ==================== Recall@K ====================

class TestRecallAtK:
    def test_recall_all_in_top3(self, relevant_ids, ranked_ids):
        """top-3 包含 3 个相关中的 2 个 → 2/3"""
        assert recall_at_k(relevant_ids, ranked_ids, 3) == pytest.approx(2 / 3)

    def test_recall_all_in_top5(self, relevant_ids, ranked_ids):
        """top-5 包含全部 3 个 → 1.0"""
        assert recall_at_k(relevant_ids, ranked_ids, 5) == pytest.approx(1.0)

    def test_recall_none_found(self, relevant_ids, no_match_ids):
        """完全不相关 → 0.0"""
        assert recall_at_k(relevant_ids, no_match_ids, 3) == pytest.approx(0.0)

    def test_recall_empty_relevant(self, empty_relevant, ranked_ids):
        """没有相关文档时 → 1.0（没有遗漏）"""
        assert recall_at_k(empty_relevant, ranked_ids, 3) == pytest.approx(1.0)

    def test_recall_empty_ranked(self, relevant_ids, empty_ranked):
        """检索结果为空 → 0.0"""
        assert recall_at_k(relevant_ids, empty_ranked, 3) == pytest.approx(0.0)

    def test_recall_k_larger_than_results(self, relevant_ids):
        """K 大于检索结果数"""
        assert recall_at_k(relevant_ids, ["a", "b"], 10) == pytest.approx(2 / 3)

    def test_recall_k_zero(self, relevant_ids, ranked_ids):
        """K=0 → 0.0"""
        assert recall_at_k(relevant_ids, ranked_ids, 0) == pytest.approx(0.0)


# ==================== Precision@K ====================

class TestPrecisionAtK:
    def test_precision_top3(self, relevant_ids, ranked_ids):
        """top-3 中 2 个相关 → 2/3"""
        assert precision_at_k(relevant_ids, ranked_ids, 3) == pytest.approx(2 / 3)

    def test_precision_top5(self, relevant_ids, ranked_ids):
        """top-5 中 3 个相关 → 3/5 = 0.6"""
        assert precision_at_k(relevant_ids, ranked_ids, 5) == pytest.approx(3 / 5)

    def test_precision_all_relevant_top(self, relevant_ids):
        """top-3 全是相关 → 1.0"""
        assert precision_at_k(relevant_ids, ["a", "b", "c"], 3) == pytest.approx(1.0)

    def test_precision_none_relevant(self, relevant_ids, no_match_ids):
        """top-3 全不相关 → 0.0"""
        assert precision_at_k(relevant_ids, no_match_ids, 3) == pytest.approx(0.0)

    def test_precision_k_zero(self, relevant_ids, ranked_ids):
        """K=0 → 0.0"""
        assert precision_at_k(relevant_ids, ranked_ids, 0) == pytest.approx(0.0)

    def test_precision_empty_ranked(self, relevant_ids, empty_ranked):
        """检索结果为空 → 0.0"""
        assert precision_at_k(relevant_ids, empty_ranked, 3) == pytest.approx(0.0)


# ==================== MRR ====================

class TestMRR:
    def test_mrr_first_relevant(self, relevant_ids, ranked_ids):
        """第一个就是相关 → 1/1 = 1.0"""
        assert mrr(relevant_ids, ranked_ids) == pytest.approx(1.0)

    def test_mrr_second_relevant(self, relevant_ids):
        """第二个是相关 → 1/2 = 0.5"""
        assert mrr(relevant_ids, ["x", "a", "y"]) == pytest.approx(0.5)

    def test_mrr_third_relevant(self, relevant_ids):
        """第三个是相关 → 1/3"""
        assert mrr(relevant_ids, ["x", "y", "a"]) == pytest.approx(1 / 3)

    def test_mrr_no_match(self, relevant_ids, no_match_ids):
        """全不相关 → 0.0"""
        assert mrr(relevant_ids, no_match_ids) == pytest.approx(0.0)

    def test_mrr_empty_relevant(self, empty_relevant, ranked_ids):
        """无相关 doc → 0.0"""
        assert mrr(empty_relevant, ranked_ids) == pytest.approx(0.0)

    def test_mrr_empty_ranked(self, relevant_ids, empty_ranked):
        """空检索 → 0.0"""
        assert mrr(relevant_ids, empty_ranked) == pytest.approx(0.0)


# ==================== NDCG@K ====================

class TestNDCGAtK:
    def test_ndcg_perfect_order(self, graded_relevance):
        """完美排序（按分数降序） → 接近 1.0"""
        ideal = ["a", "b", "c", "x", "y", "z"]
        score = ndcg_at_k(graded_relevance, ideal, 3)
        assert score > 0.95

    def test_ndcg_reversed_order(self, graded_relevance):
        """倒序 → 低于完美排序"""
        reversed_order = ["c", "b", "a"]
        score_rev = ndcg_at_k(graded_relevance, reversed_order, 3)
        # 倒序应该比完美排序差
        perfect = ndcg_at_k(graded_relevance, ["a", "b", "c"], 3)
        assert score_rev < perfect

    def test_ndcg_empty_ranked(self, graded_relevance, empty_ranked):
        """空检索 → 0.0"""
        assert ndcg_at_k(graded_relevance, empty_ranked, 3) == pytest.approx(0.0)

    def test_ndcg_all_zeros(self):
        """全部得分为 0 → 0.0"""
        rel = {"x": 0.0, "y": 0.0}
        assert ndcg_at_k(rel, ["x", "y"], 2) == pytest.approx(0.0)

    def test_ndcg_single_item(self):
        """单条结果"""
        rel = {"a": 5.0}
        score = ndcg_at_k(rel, ["a"], 1)
        assert score == pytest.approx(1.0)


# ==================== Hit Rate@K ====================

class TestHitRate:
    def test_hit_at_1(self, relevant_ids, ranked_ids):
        """第 1 个就命中 → 1.0"""
        assert hit_rate(relevant_ids, ranked_ids, 1) == pytest.approx(1.0)

    def test_hit_at_3(self, relevant_ids, ranked_ids):
        """top-3 内命中 → 1.0"""
        assert hit_rate(relevant_ids, ranked_ids, 3) == pytest.approx(1.0)

    def test_hit_miss(self, relevant_ids, no_match_ids):
        """全不相关 → 0.0"""
        assert hit_rate(relevant_ids, no_match_ids, 3) == pytest.approx(0.0)

    def test_hit_empty_relevant(self, empty_relevant, ranked_ids):
        """空相关集合 → 0.0"""
        assert hit_rate(empty_relevant, ranked_ids, 3) == pytest.approx(0.0)

    def test_hit_empty_ranked(self, relevant_ids, empty_ranked):
        """空检索 → 0.0"""
        assert hit_rate(relevant_ids, empty_ranked, 3) == pytest.approx(0.0)

    def test_hit_k_zero(self, relevant_ids, ranked_ids):
        """K=0 → 0.0"""
        assert hit_rate(relevant_ids, ranked_ids, 0) == pytest.approx(0.0)


# ==================== MAP ====================

class TestAveragePrecision:
    def test_map_perfect(self, relevant_ids):
        """完美排序 → 1.0"""
        assert average_precision(relevant_ids, ["a", "b", "c"]) == pytest.approx(1.0)

    def test_map_with_noise(self, relevant_ids, ranked_ids):
        """有噪声的排序"""
        # a(x=1): prec=1/1=1, b(x=3): prec=2/3, c(x=5): prec=3/5=0.6
        # map = (1 + 2/3 + 3/5) / 3 ≈ 0.7556
        result = average_precision(relevant_ids, ranked_ids)
        assert result == pytest.approx(0.7556, abs=0.001)

    def test_map_no_match(self, relevant_ids, no_match_ids):
        """全不相关 → 0.0"""
        assert average_precision(relevant_ids, no_match_ids) == pytest.approx(0.0)

    def test_map_empty_relevant(self, empty_relevant, ranked_ids):
        """无相关文档 → 0.0"""
        assert average_precision(empty_relevant, ranked_ids) == pytest.approx(0.0)

    def test_map_empty_ranked(self, relevant_ids, empty_ranked):
        """空检索 → 0.0"""
        assert average_precision(relevant_ids, empty_ranked) == pytest.approx(0.0)


# ==================== compute_all_retrieval_metrics ====================

class TestComputeAllMetrics:
    def test_returns_all_expected_keys(self, relevant_ids, ranked_ids):
        """验证返回字典包含所有预期指标"""
        result = compute_all_retrieval_metrics(relevant_ids, ranked_ids, [1, 3, 5])
        expected_keys = [
            "recall@1", "precision@1", "ndcg@1", "hit@1",
            "recall@3", "precision@3", "ndcg@3", "hit@3",
            "recall@5", "precision@5", "ndcg@5", "hit@5",
            "mrr", "map",
        ]
        for key in expected_keys:
            assert key in result, f"缺少指标: {key}"
        assert len(result) == len(expected_keys)

    def test_all_values_in_range(self, relevant_ids, ranked_ids):
        """所有指标值在 [0, 1] 范围内"""
        result = compute_all_retrieval_metrics(relevant_ids, ranked_ids, [1, 3, 5, 8])
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key} = {val} 超出 [0,1] 范围"

    def test_perfect_retrieval_all_ones(self, relevant_ids):
        """完美检索：所有相关文档排在前面，核心指标符合预期"""
        result = compute_all_retrieval_metrics(
            relevant_ids, ["a", "b", "c", "x"], [1, 3]
        )
        # Hit: 排名1位置就是相关文档 → hit@1=1.0, hit@3=1.0
        assert result["hit@1"] == 1.0, "hit@1 应为 1.0"
        assert result["hit@3"] == 1.0, "hit@3 应为 1.0"
        # Recall@1: 只有 1/3 个相关文档在 top-1
        assert result["recall@1"] == pytest.approx(1 / 3), f"recall@1 应为 1/3"
        # Recall@3: 3 个相关全在 top-3
        assert result["recall@3"] == 1.0, f"recall@3 应为 1.0"
        # Precision@1: top-1 是相关文档
        assert result["precision@1"] == 1.0, f"precision@1 应为 1.0"
        # MRR: 第一个就是相关
        assert result["mrr"] == 1.0, f"mrr 应为 1.0"

    def test_failed_retrieval_all_zeros(self, relevant_ids, no_match_ids):
        """全不相关的检索结果：指标应接近 0"""
        result = compute_all_retrieval_metrics(relevant_ids, no_match_ids, [1, 3])
        for key in result:
            if key.startswith("recall") or key.startswith("hit"):
                assert result[key] == 0.0, f"{key} 应为 0.0"

    def test_with_graded_relevance(self, relevant_ids, ranked_ids, graded_relevance):
        """使用分级评分计算"""
        result = compute_all_retrieval_metrics(
            relevant_ids, ranked_ids, [3], graded_relevance
        )
        assert "ndcg@3" in result


# ==================== 边界与极端情况 ====================

class TestEdgeCases:
    def test_single_relevant_single_retrieved(self):
        """单文档场景"""
        assert recall_at_k({"a"}, ["a"], 1) == pytest.approx(1.0)
        assert precision_at_k({"a"}, ["a"], 1) == pytest.approx(1.0)
        assert mrr({"a"}, ["a"]) == pytest.approx(1.0)
        assert hit_rate({"a"}, ["a"], 1) == pytest.approx(1.0)

    def test_single_relevant_not_retrieved(self):
        """单文档未命中"""
        assert recall_at_k({"a"}, ["x"], 1) == pytest.approx(0.0)
        assert precision_at_k({"a"}, ["x"], 1) == pytest.approx(0.0)

    def test_duplicate_ids(self):
        """重复 ID"""
        rel = {"a", "b"}
        ranked = ["a", "a", "b"]  # a 出现两次
        # precision@3: 出现两次的 "a" 都算相关，但 "a" 在 set 中只算一次
        # hits = 1(a) + 1(a) + 1(b) = 3, precision = 3/3 = 1.0
        assert precision_at_k(rel, ranked, 3) == pytest.approx(1.0)

    def test_large_k(self, relevant_ids, ranked_ids):
        """K 远大于检索结果数"""
        assert recall_at_k(relevant_ids, ranked_ids, 100) == pytest.approx(1.0)
        assert hit_rate(relevant_ids, ranked_ids, 100) == pytest.approx(1.0)
