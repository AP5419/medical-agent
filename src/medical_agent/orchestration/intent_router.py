# -*- coding: utf-8 -*-
"""意图路由器 - 第3层编排层：分析用户消息意图并检测紧急情况"""

from dataclasses import dataclass, field
from enum import Enum
import re
import json
from typing import Optional
from loguru import logger


class IntentType(str, Enum):
    """用户意图类型枚举"""
    INQUIRY = "inquiry"         # 分诊导诊
    REPORT = "report"           # 报告解读
    DRUG = "drug"               # 药物咨询
    KNOWLEDGE = "knowledge"     # 知识问答
    OPERATION = "operation"     # 运营数据
    GREETING = "greeting"       # 问候寒暄


@dataclass
class IntentResult:
    """意图识别结果"""
    intent: IntentType
    confidence: float = 0.5
    is_emergency: bool = False
    emergency_keywords: list = field(default_factory=list)


# 紧急情况关键词列表（20+个中文紧急症状关键词）
EMERGENCY_KEYWORDS = [
    "胸痛", "呼吸困难", "意识不清", "昏迷", "大出血",
    "中风", "心肌梗死", "心脏骤停", "窒息", "抽搐",
    "休克", "严重过敏", "急性腹痛", "高烧不退", "吐血",
    "咯血", "剧烈头痛", "视力突然丧失", "肢体瘫痪", "言语不清",
    "中毒", "溺水", "电击", "坠落", "车祸重伤",
    "烧伤", "冻伤", "自缢",
]

# 编译紧急情况正则模式（忽略大小写）
_EMERGENCY_PATTERNS = [re.compile(re.escape(kw), re.IGNORECASE) for kw in EMERGENCY_KEYWORDS]

# 问候语检测关键词
_GREETING_KEYWORDS = [
    "你好", "hi", "hello", "嗨", "早上好", "晚上好", "下午好",
    "谢谢", "感谢", "再见", "拜拜", "bye",
]

# 关键词意图预筛（命中则跳过 LLM，节省 ~14s/轮）
_INTENT_KEYWORD_MAP = {
    # 药品咨询关键词
    IntentType.DRUG: [
        "药品", "药物", "吃药", "用药", "剂量", "用法", "用量",
        "副作用", "禁忌", "说明书", "处方", "抗生素", "消炎药",
        "降压药", "降糖药", "阿司匹林", "头孢", "阿莫西林",
        "服用", "口服", "外用药", "注射", "皮试",
    ],
    # 报告解读关键词
    IntentType.REPORT: [
        "报告", "化验", "检查结果", "化验单", "体检单", "报告单",
        "CT", "MRI", "B超", "彩超", "X光", "心电图", "血常规",
        "尿常规", "肝功能", "肾功能", "血糖", "血脂", "血压",
        "指标", "偏高", "偏低", "异常", "复查",
        "信息", "档案", "记录", "病历", "就诊记录", "出院", "入院",
    ],
    # 知识问答关键词
    IntentType.KNOWLEDGE: [
        "什么是", "是什么病", "怎么回事", "为什么会", "如何治疗",
        "治疗方法", "怎么治疗", "怎么治", "怎么办", "严重吗",
        "会传染吗", "能治好吗", "科普", "指南", "预防",
    ],
    # 运营数据关键词
    IntentType.OPERATION: [
        "统计", "报表", "运营", "数据", "KPI", "业绩",
        "看板", "图表", "导出", "分析报表", "数据统计",
    ],
    # 症状/不适关键词（最常命中）
    IntentType.INQUIRY: [
        "不舒服", "难受", "疼", "痛", "症状", "生病",
        "挂号", "看什么科", "挂什么科", "哪个科室", "就诊",
        "发烧", "咳嗽", "恶心", "头晕", "腹泻", "呕吐",
        "乏力", "胸闷", "心慌", "气短", "肚子", "胃",
        "头痛", "嗓子", "喉咙", "流鼻涕", "鼻塞", "拉肚子",
    ],
}

# 意图分类提示词
INTENT_CLASSIFY_PROMPT = """你是一个医疗意图分类器。从以下6个意图中选择最匹配的一个。

意图类型及区分规则（必须选一个，不能返回其他值）：
- inquiry: 分诊导诊——用户描述身体症状（疼痛/不适/发热/咳嗽等），询问挂什么科室或就医建议
- drug: 药物咨询——用户提及任何药品相关话题："XX可以吃吗""怎么吃""副作用""用法用量"
  注意：消息中提到任何药品名称（中文/英文/商品名）优先选 drug；"可以吃吗"结构选 drug 而非 inquiry
- report: 报告解读——用户提及化验单、检查报告、血常规、CT、B超、指标异常、出院等
- knowledge: 知识问答——用户询问医学知识科普："什么是XX病""为什么会XX""如何治疗"
- operation: 运营数据——管理员查询统计数据、报表、KPI
- greeting: 问候寒暄——打招呼、感谢、告别

请返回JSON格式：{{"intent": "意图类型", "confidence": 0.0-1.0}}

用户消息：{message}

仅返回JSON，不要其他内容。"""


class IntentRouter:
    """意图路由器——分析用户消息，检测紧急情况，分类意图"""

    def __init__(self):
        """初始化意图路由器，LLM延迟加载"""
        self._llm: Optional[object] = None

    def _ensure_llm(self):
        """延迟导入并创建LLM实例"""
        if self._llm is None:
            from medical_agent.providers.llm import get_llm_qa
            self._llm = get_llm_qa()

    def detect_emergency(self, message: str) -> tuple[bool, list]:
        """通过正则匹配检测紧急情况

        Args:
            message: 用户输入消息

        Returns:
            (is_emergency, matched_keywords): 是否为紧急情况及匹配到的关键词列表
        """
        matched = []
        for pattern in _EMERGENCY_PATTERNS:
            if pattern.search(message):
                # 从原始EMERGENCY_KEYWORDS中找到对应的关键词
                for kw in EMERGENCY_KEYWORDS:
                    if re.search(re.escape(kw), message, re.IGNORECASE):
                        if kw not in matched:
                            matched.append(kw)
        return len(matched) > 0, matched

    async def classify(self, message: str) -> IntentResult:
        """分类用户消息意图

        Args:
            message: 用户输入消息

        Returns:
            IntentResult: 包含意图类型、置信度、紧急标志的结果
        """
        # 第一步：检测紧急情况
        is_emergency, keywords = self.detect_emergency(message)
        if is_emergency:
            return IntentResult(
                intent=IntentType.INQUIRY,
                confidence=1.0,
                is_emergency=True,
                emergency_keywords=keywords,
            )

        # 第二步：简单问候检测
        msg_lower = message.lower().strip()
        msg_len = len(message.strip())
        if any(kw in msg_lower for kw in _GREETING_KEYWORDS) and msg_len < 30:
            return IntentResult(
                intent=IntentType.GREETING,
                confidence=0.9,
            )

        # 第三步：关键词预筛（命中则跳过 LLM，节省 ~14s）
        for intent_type, keywords in _INTENT_KEYWORD_MAP.items():
            for kw in keywords:
                if kw in msg_lower:
                    logger.info(f"[意图] 关键词命中: '{kw}' → {intent_type.value}")
                    return IntentResult(intent=intent_type, confidence=0.85)

        # 第四步：LLM意图分类
        logger.info(f"[意图] 关键词未命中，走LLM分类: '{message[:60]}'")
        try:
            self._ensure_llm()
            prompt = INTENT_CLASSIFY_PROMPT.format(message=message)
            response = await self._llm.ainvoke(prompt)
            text = response.content.strip()

            # 提取JSON
            json_match = re.search(r'\{[^{}]*\}', text)
            if json_match:
                data = json.loads(json_match.group())
                intent_str = data.get("intent", "inquiry")
                confidence = float(data.get("confidence", 0.5))

                # 映射到IntentType
                intent_map = {
                    "inquiry": IntentType.INQUIRY,
                    "report": IntentType.REPORT,
                    "drug": IntentType.DRUG,
                    "knowledge": IntentType.KNOWLEDGE,
                    "operation": IntentType.OPERATION,
                    "greeting": IntentType.GREETING,
                }
                intent = intent_map.get(intent_str, IntentType.INQUIRY)
                return IntentResult(intent=intent, confidence=confidence)

        except Exception as e:
            logger.warning(f"[意图] LLM分类失败: {e}")

        # 默认返回INQUIRY
        logger.info(f"[意图] 分类结果: inquiry (default)")
        return IntentResult(intent=IntentType.INQUIRY, confidence=0.3)
