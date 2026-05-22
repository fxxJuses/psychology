"""
检索评估指标 —— 纯函数实现，仅依赖 numpy。

所有函数均不涉及 LLM 调用，可独立进行单元测试。
"""

import numpy as np
from typing import List, Set, Dict


def recall_at_k(
    relevant_ids: Set[str],
    retrieved_ids: List[str],
    k: int,
) -> float:
    """
    计算 Recall@K：top-K 检索结果中命中的相关文档占全部相关文档的比例。

    Args:
        relevant_ids: 所有相关文档的 ID 集合
        retrieved_ids: 检索返回的文档 ID 列表（按排名排序）
        k: 截断值

    Returns:
        float: 0.0 ~ 1.0 之间的召回率
    """
    if len(relevant_ids) == 0:
        return 1.0  # 如果没有相关文档，认为召回率为 1（没有遗漏）
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(
    relevant_ids: Set[str],
    retrieved_ids: List[str],
    k: int,
) -> float:
    """
    计算 Precision@K：top-K 检索结果中相关文档的占比。

    Args:
        relevant_ids: 所有相关文档的 ID 集合
        retrieved_ids: 检索返回的文档 ID 列表（按排名排序）
        k: 截断值

    Returns:
        float: 0.0 ~ 1.0 之间的精确率
    """
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if len(top_k) == 0:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(top_k)


def mrr(relevant_ids: Set[str], ranked_ids: List[str]) -> float:
    """
    计算 MRR（Mean Reciprocal Rank）：第一个相关文档排名倒数的平均值。

    MRR = 1 / rank_of_first_relevant
    如果没有任何相关文档命中，返回 0。

    Args:
        relevant_ids: 所有相关文档的 ID 集合
        ranked_ids: 检索返回的文档 ID 列表（按排名排序）

    Returns:
        float: 0.0 ~ 1.0 之间的 MRR 值
    """
    for rank, rid in enumerate(ranked_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    graded_relevance: Dict[str, float],
    ranked_ids: List[str],
    k: int,
) -> float:
    """
    计算 NDCG@K（Normalized Discounted Cumulative Gain）。

    使用分级相关性评分（如 0=无关, 1=弱相关, 2=相关, 3=高度相关），
    考虑排名位置的重要性（排名越靠前，权重越高）。

    Args:
        graded_relevance: {doc_id: relevance_score} 映射，分数越高越相关
        ranked_ids: 检索返回的文档 ID 列表（按排名排序）
        k: 截断值

    Returns:
        float: 0.0 ~ 1.0 之间的 NDCG 值
    """
    top_k = ranked_ids[:k]
    if len(top_k) == 0:
        return 0.0

    # DCG
    dcg = 0.0
    for i, rid in enumerate(top_k):
        rel = graded_relevance.get(rid, 0.0)
        # 使用 log2(i+2) 作为折扣因子，第一个位置 i=0 → log2(2)=1
        dcg += rel / np.log2(i + 2)

    # IDCG（理想排序：所有相关文档按分数降序排列）
    all_relevances = sorted(graded_relevance.values(), reverse=True)
    idcg = 0.0
    for i in range(min(k, len(all_relevances))):
        idcg += all_relevances[i] / np.log2(i + 2)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def hit_rate(
    relevant_ids: Set[str],
    retrieved_ids: List[str],
    k: int,
) -> float:
    """
    计算 Hit Rate@K：top-K 中是否至少有一个相关文档。

    Args:
        relevant_ids: 所有相关文档的 ID 集合
        retrieved_ids: 检索返回的文档 ID 列表（按排名排序）
        k: 截断值

    Returns:
        float: 1.0（命中）或 0.0（未命中）
    """
    if len(relevant_ids) == 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k & relevant_ids else 0.0


def average_precision(
    relevant_ids: Set[str],
    ranked_ids: List[str],
) -> float:
    """
    计算 Average Precision：每个相关文档位置上的 Precision 的平均值。

    Args:
        relevant_ids: 所有相关文档的 ID 集合
        ranked_ids: 检索返回的文档 ID 列表（按排名排序）

    Returns:
        float: 0.0 ~ 1.0 之间的 Average Precision 值
    """
    if len(relevant_ids) == 0:
        return 0.0

    hits = 0
    sum_prec = 0.0
    for rank, rid in enumerate(ranked_ids, start=1):
        if rid in relevant_ids:
            hits += 1
            sum_prec += hits / rank

    return sum_prec / len(relevant_ids)


def compute_all_retrieval_metrics(
    relevant_ids: Set[str],
    ranked_ids: List[str],
    k_values: List[int] = None,
    graded_relevance: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    批量计算所有检索指标。

    Args:
        relevant_ids: 相关文档 ID 集合
        ranked_ids: 检索结果 ID 列表（按排名）
        k_values: K 值列表，默认 [1, 3, 5, 8]
        graded_relevance: 分级相关性评分，用于 NDCG。若未提供，则用相关/不相关二分。

    Returns:
        Dict[str, float]: 各项指标得分
    """
    if k_values is None:
        k_values = [1, 3, 5, 8]

    if graded_relevance is None:
        # 用二分相关性构造 graded_relevance
        graded_relevance = {rid: 1.0 for rid in relevant_ids}

    results = {}
    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(relevant_ids, ranked_ids, k)
        results[f"precision@{k}"] = precision_at_k(relevant_ids, ranked_ids, k)
        results[f"ndcg@{k}"] = ndcg_at_k(graded_relevance, ranked_ids, k)
        results[f"hit@{k}"] = hit_rate(relevant_ids, ranked_ids, k)

    results["mrr"] = mrr(relevant_ids, ranked_ids)
    results["map"] = average_precision(relevant_ids, ranked_ids)

    return results
