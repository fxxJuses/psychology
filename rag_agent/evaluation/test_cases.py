"""
RAG 评估测试用例定义。

包含 TestCase 数据类和 TestSuite 测试集，内置 12-15 条中文精选用例，
覆盖 5 种查询类型 + 负样本。
"""

from dataclasses import dataclass, field
from typing import List, Optional


# 查询类别常量
CATEGORY_FACTUAL = "factual_lookup"    # 事实检索
CATEGORY_DEFINITION = "definition"     # 概念定义
CATEGORY_COMPARISON = "comparison"     # 对比分析
CATEGORY_MULTI_HOP = "multi_hop"       # 多跳推理
CATEGORY_SUMMARY = "summary"           # 总结归纳
CATEGORY_NEGATIVE = "negative"         # 负样本（语料库中不存在）


@dataclass
class TestCase:
    """单个评估测试用例"""
    id: str                              # 唯一标识，如 "tc_001"
    category: str                        # 查询类别
    question: str                        # 用户问题
    reference_answer: str                # 参考答案（用于 Faithfulness / Recall 评估）
    expected_keywords: List[str] = field(default_factory=list)   # 期望出现的关键词
    relevant_sources: List[str] = field(default_factory=list)    # 答案应出自哪些文档
    description: str = ""                # 用例说明


class TestSuite:
    """测试用例集"""

    def __init__(self, name: str = "default"):
        self.name = name
        self.test_cases: List[TestCase] = []

    def add_case(self, case: TestCase) -> None:
        self.test_cases.append(case)

    def get_by_category(self, category: str) -> List[TestCase]:
        return [tc for tc in self.test_cases if tc.category == category]

    @property
    def categories(self) -> List[str]:
        return sorted(set(tc.category for tc in self.test_cases))

    def __len__(self) -> int:
        return len(self.test_cases)

    def __iter__(self):
        return iter(self.test_cases)

    @classmethod
    def load_default(cls) -> "TestSuite":
        """加载内置的默认测试集（12-15 条精选用例）"""
        suite = cls(name="default")

        # ==================== 事实检索 (factual_lookup) ====================
        suite.add_case(TestCase(
            id="tc_fact_001",
            category=CATEGORY_FACTUAL,
            question="广泛性焦虑症（GAD）在DSM-5中的诊断标准是什么？",
            reference_answer="根据DSM-5，广泛性焦虑症的诊断标准包括：对多种事件或活动过度焦虑和担忧，持续时间至少6个月，且难以控制这种担忧。此外还需伴随至少三项症状：坐立不安、容易疲劳、注意力难以集中、易怒、肌肉紧张、睡眠障碍。",
            expected_keywords=["DSM-5", "6个月", "过度焦虑", "诊断标准", "睡眠障碍"],
            relevant_sources=["焦虑心理学"],
            description="测试对焦虑症诊断标准的精确检索能力",
        ))
        suite.add_case(TestCase(
            id="tc_fact_002",
            category=CATEGORY_FACTUAL,
            question="社交焦虑障碍的典型表现有哪些？",
            reference_answer="社交焦虑障碍的典型表现包括：对社交场合或表演情境的显著恐惧或焦虑，担心自己的行为会让自己难堪或被他人负面评价。生理症状包括脸红、出汗、颤抖、心跳加速。患者通常会回避社交场合，或带着强烈的焦虑勉强忍受。",
            expected_keywords=["社交焦虑", "恐惧", "回避", "负面评价", "脸红", "出汗"],
            relevant_sources=["焦虑心理学"],
            description="测试对社交焦虑具体症状的检索",
        ))
        suite.add_case(TestCase(
            id="tc_fact_003",
            category=CATEGORY_FACTUAL,
            question="勒庞在《乌合之众》中描述了群体的哪些主要特征？",
            reference_answer="勒庞认为群体具有以下主要特征：冲动易变、易受暗示和轻信、情绪夸张和简单化、偏狭专横和保守。群体中的个人会丧失独立人格，思想和情感通过暗示和传染转向同一方向。",
            expected_keywords=["冲动", "暗示", "传染", "群体", "轻信", "人格"],
            relevant_sources=["乌合之众：大众心理研究"],
            description="测试对乌合之众核心概念的检索",
        ))

        # ==================== 概念定义 (definition) ====================
        suite.add_case(TestCase(
            id="tc_def_001",
            category=CATEGORY_DEFINITION,
            question="什么是心理学中的'习得性无助'？",
            reference_answer="习得性无助是指个体在反复经历无法控制的负面事件后，形成的一种消极、被动的心理状态。即使有机会改变处境，个体也会放弃尝试。这一概念由心理学家马丁·塞利格曼通过动物实验提出。",
            expected_keywords=["塞利格曼", "消极", "被动", "无法控制", "放弃"],
            relevant_sources=["人人都该懂的心理学"],
            description="测试对经典心理学概念的检索和理解",
        ))
        suite.add_case(TestCase(
            id="tc_def_002",
            category=CATEGORY_DEFINITION,
            question="什么是'群体极化'现象？",
            reference_answer="群体极化是指群体讨论会使成员最初的倾向得到加强，使得群体的决策比个体成员的初始倾向更为极端。如果群体成员最初倾向于冒险，讨论后会更冒险；如果最初倾向于保守，讨论后会更保守。",
            expected_keywords=["群体讨论", "极端", "倾向加强", "冒险", "保守"],
            relevant_sources=["人人都该懂的心理学"],
            description="测试对群体心理学概念的检索",
        ))
        suite.add_case(TestCase(
            id="tc_def_003",
            category=CATEGORY_DEFINITION,
            question="在焦虑心理学中，什么是'灾难化思维'？",
            reference_answer="灾难化思维是一种常见的认知歪曲，指个体倾向于将小问题夸大为灾难性事件，总是预测最坏的结果。例如把轻微的身体不适解读为严重疾病的征兆，把工作上的小失误看作职业毁灭的前兆。",
            expected_keywords=["认知歪曲", "灾难", "最坏结果", "夸大"],
            relevant_sources=["焦虑心理学"],
            description="测试对焦虑认知模式的检索",
        ))

        # ==================== 对比分析 (comparison) ====================
        suite.add_case(TestCase(
            id="tc_cmp_001",
            category=CATEGORY_COMPARISON,
            question="焦虑和恐惧在心理学定义上有哪些核心区别？",
            reference_answer="恐惧是对当前明确、具体的威胁产生的情绪反应，如面对猛兽时感到害怕。焦虑则是对未来不确定的、潜在的威胁产生的担忧，往往没有明确对象。恐惧是即时的生理警报反应，焦虑是持续的认知担忧状态。",
            expected_keywords=["恐惧", "具体威胁", "未来", "不确定", "即时", "持续"],
            relevant_sources=["焦虑心理学"],
            description="测试对两个易混淆概念的区分能力",
        ))
        suite.add_case(TestCase(
            id="tc_cmp_002",
            category=CATEGORY_COMPARISON,
            question="精神分析学派和行为主义学派对焦虑的解释有什么不同？",
            reference_answer="精神分析学派认为焦虑源于潜意识中的冲突，特别是本我、自我和超我之间的张力。而行为主义学派认为焦虑是通过条件反射习得的，是将中性刺激与恐惧刺激关联后的产物，可通过系统脱敏等方式消除。",
            expected_keywords=["潜意识", "冲突", "条件反射", "习得", "系统脱敏"],
            relevant_sources=["人人都该懂的心理学", "焦虑心理学"],
            description="测试跨文档的对比分析能力",
        ))

        # ==================== 多跳推理 (multi_hop) ====================
        suite.add_case(TestCase(
            id="tc_mh_001",
            category=CATEGORY_MULTI_HOP,
            question="认知行为疗法（CBT）的核心技术有哪些？它们如何应用于焦虑症的治疗？",
            reference_answer="CBT核心技术包括认知重构、行为实验、暴露疗法和放松训练。认知重构帮助患者识别并挑战灾难化思维和负性自动思维。暴露疗法让患者逐步面对焦虑源，打破回避行为。放松训练包括腹式呼吸和渐进性肌肉放松，缓解焦虑的生理症状。",
            expected_keywords=["认知重构", "暴露疗法", "放松训练", "自动思维", "回避"],
            relevant_sources=["焦虑心理学", "人人都该懂的心理学"],
            description="测试需要综合多个chunk信息的能力",
        ))
        suite.add_case(TestCase(
            id="tc_mh_002",
            category=CATEGORY_MULTI_HOP,
            question="从众心理产生的原因是什么？它和群体暗示有什么关联？",
            reference_answer="从众心理产生的原因包括信息性社会影响和规范性社会影响。信息性影响指个体在不确定情况下参考他人的行为作为信息源；规范性影响指个体为获得群体接纳而顺从。这与群体暗示的关联在于，群体中通过暗示和传染机制，使个体更容易放弃独立判断。",
            expected_keywords=["信息性影响", "规范性影响", "暗示", "传染", "独立判断"],
            relevant_sources=["乌合之众：大众心理研究", "人人都该懂的心理学"],
            description="测试跨文档、多概念的推理能力",
        ))

        # ==================== 总结归纳 (summary) ====================
        suite.add_case(TestCase(
            id="tc_sum_001",
            category=CATEGORY_SUMMARY,
            question="《乌合之众》这本书的核心论点是什么？请简要概括。",
            reference_answer="《乌合之众》的核心论点是：当个人融入群体后，会丧失理性思考和独立人格，情感和思想统一趋向于同一方向，形成一种'集体心理'。群体具有冲动、易受暗示、情绪夸张和偏执的特征。勒庞认为群体力量的崛起标志着大众时代的到来。",
            expected_keywords=["集体心理", "丧失理性", "群体", "暗示", "大众时代"],
            relevant_sources=["乌合之众：大众心理研究"],
            description="测试对整本书核心论点的总结归纳能力（已知RAG在全局理解上有局限）",
        ))
        suite.add_case(TestCase(
            id="tc_sum_002",
            category=CATEGORY_SUMMARY,
            question="常见的焦虑症类型有哪些？它们各自的核心特征是什么？",
            reference_answer="常见的焦虑症类型包括：广泛性焦虑症（持续过度担忧）、社交焦虑障碍（对社交场合的恐惧和回避）、惊恐障碍（反复发作的惊恐发作）、广场恐怖症（对开放空间或拥挤场所的恐惧）、特定恐怖症（对特定物品或情境的极端恐惧）、分离焦虑障碍（与依恋对象分离时的过度焦虑）。",
            expected_keywords=["广泛性焦虑", "社交焦虑", "惊恐障碍", "恐怖症", "分离焦虑"],
            relevant_sources=["焦虑心理学"],
            description="测试对分类信息的综合归纳能力",
        ))

        # ==================== 负样本 (negative) ====================
        suite.add_case(TestCase(
            id="tc_neg_001",
            category=CATEGORY_NEGATIVE,
            question="马斯洛需求层次理论中的最高层次是什么？",
            reference_answer="文档中未找到相关信息。",
            expected_keywords=[],
            relevant_sources=[],
            description="负样本：马斯洛理论不在三本心理学书的覆盖范围内，应返回'未找到'",
        ))
        suite.add_case(TestCase(
            id="tc_neg_002",
            category=CATEGORY_NEGATIVE,
            question="Python语言中如何实现多线程并发？",
            reference_answer="文档中未找到相关信息。",
            expected_keywords=[],
            relevant_sources=[],
            description="负样本：完全无关的技术问题，测试检索器的噪声过滤能力",
        ))
        suite.add_case(TestCase(
            id="tc_neg_003",
            category=CATEGORY_NEGATIVE,
            question="2024年诺贝尔物理学奖颁给了谁？",
            reference_answer="文档中未找到相关信息。",
            expected_keywords=[],
            relevant_sources=[],
            description="负样本：时效性问题，三本心理学书中不包含此类信息",
        ))

        return suite
