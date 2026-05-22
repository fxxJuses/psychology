"""
评估结果报告 —— 格式化控制台输出和 JSON 导出。
"""

import json
from typing import Dict, List

from .runner import EvalResult


class EvalReporter:
    """评估结果报告器"""

    @staticmethod
    def print_retrieval_table(
        results: Dict[str, Dict[str, float]],
        title: str = "检索评估结果",
    ) -> None:
        """
        打印检索指标对比表。

        Args:
            results: {mode: {metric_name: avg_score}}
            title: 表格标题
        """
        if not results:
            print("\n(无检索评估数据)")
            return

        modes = list(results.keys())

        # 收集所有指标名（按固定顺序排列）
        metric_order = []
        for k_val in [1, 3, 5, 8]:
            metric_order.extend([f"recall@{k_val}", f"precision@{k_val}", f"ndcg@{k_val}", f"hit@{k_val}"])
        metric_order.extend(["mrr", "map"])

        # 只保留存在的指标
        all_metrics = set()
        for mode_results in results.values():
            all_metrics.update(mode_results.keys())
        metric_order = [m for m in metric_order if m in all_metrics]

        if not metric_order:
            return

        print(f"\n{'─' * 80}")
        print(f"  {title}")
        print(f"{'─' * 80}")

        # 表头
        header = f"  {'指标':<18}"
        for mode in modes:
            header += f" {mode:>12}"
        print(header)
        print(f"  {'─' * 18}{'─' * (13 * len(modes))}")

        # 逐行输出
        for metric in metric_order:
            # 格式化指标名
            display_name = metric.replace("@", "@").replace("_", " ")
            row = f"  {display_name:<18}"
            for mode in modes:
                score = results[mode].get(metric, 0.0)
                # 颜色标记：>=0.7 好, >=0.4 中, <0.4 差
                row += f" {score:>11.4f}"
            print(row)

        print(f"{'─' * 80}")

    @staticmethod
    def print_generation_table(results: Dict[str, float]) -> None:
        """打印生成质量评估表"""
        if not results:
            print("\n(无生成评估数据)")
            return

        print(f"\n{'─' * 50}")
        print(f"  生成质量评估结果")
        print(f"{'─' * 50}")
        for metric_name, score in results.items():
            name_display = {
                "faithfulness": "忠实度 (Faithfulness)",
                "answer_relevance": "答案相关性 (Answer Relevance)",
            }.get(metric_name, metric_name)
            print(f"  {name_display:<30} {score:>8.4f}")
        print(f"{'─' * 50}")

    @staticmethod
    def print_category_table(
        per_category: Dict[str, Dict[str, Dict[str, float]]],
        metric: str = "recall@5",
    ) -> None:
        """按类别打印指定指标的对比表"""
        if not per_category:
            return

        modes = list(next(iter(per_category.values())).keys())

        print(f"\n{'─' * 80}")
        print(f"  按类别对比 —— {metric}")
        print(f"{'─' * 80}")

        header = f"  {'类别':<20}"
        for mode in modes:
            header += f" {mode:>12}"
        print(header)
        print(f"  {'─' * 20}{'─' * (13 * len(modes))}")

        for cat, mode_results in per_category.items():
            row = f"  {cat:<20}"
            for mode in modes:
                score = mode_results.get(mode, {}).get(metric, 0.0)
                row += f" {score:>11.4f}"
            print(row)

        print(f"{'─' * 80}")

    @staticmethod
    def print_full_report(
        eval_result: EvalResult,
        show_per_category: bool = True,
    ) -> None:
        """
        打印完整评估报告。

        Args:
            eval_result: 评估结果
            show_per_category: 是否显示按类别拆分
        """
        print("\n" + "=" * 60)
        print("  评估报告")
        print("=" * 60)

        # 检索评估
        if eval_result.retrieval:
            EvalReporter.print_retrieval_table(eval_result.retrieval)

            # 各模式简要总结
            print(f"\n  ── 各模式概要 ──")
            for mode, metrics in eval_result.retrieval.items():
                recall5 = metrics.get("recall@5", 0.0)
                mrr_val = metrics.get("mrr", 0.0)
                hit5 = metrics.get("hit@5", 0.0)
                print(
                    f"  {mode:<10}  Recall@5={recall5:.3f}  "
                    f"MRR={mrr_val:.3f}  Hit@5={hit5:.3f}"
                )

        # 生成评估
        if eval_result.generation:
            EvalReporter.print_generation_table(eval_result.generation)

        # 按类别拆分
        if show_per_category and eval_result.per_category:
            EvalReporter.print_category_table(
                eval_result.per_category, metric="recall@5"
            )
            EvalReporter.print_category_table(
                eval_result.per_category, metric="mrr"
            )

        print("\n" + "=" * 60)

    @staticmethod
    def export_json(eval_result: EvalResult, filepath: str) -> None:
        """
        导出评估结果为 JSON 文件。

        Args:
            eval_result: 评估结果
            filepath: 输出文件路径
        """
        data = {
            "test_suite": eval_result.test_suite_name,
            "retrieval": eval_result.retrieval,
            "generation": eval_result.generation,
            "per_category": eval_result.per_category,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n评估结果已导出到: {filepath}")
