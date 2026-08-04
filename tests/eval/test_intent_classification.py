# -*- coding: utf-8 -*-
"""
意图分类评估
指标: Top-1 准确率（分 L2关键词 和 L3 LLM兜底 分别统计）
"""

import os
import pytest
from collections import Counter
from medical_agent.orchestration.intent_router import IntentRouter, _GREETING_KEYWORDS, _INTENT_KEYWORD_MAP

from tests.eval.conftest import load_dataset


def _is_keyword_hit(message: str) -> tuple[bool, str]:
    """判断消息是否命中关键词预筛层（L2），返回(是否命中, 命中意图)"""
    msg_lower = message.lower().strip()
    if len(message.strip()) < 30 and any(k in msg_lower for k in _GREETING_KEYWORDS):
        return True, "greeting"
    for intent_type, keywords in _INTENT_KEYWORD_MAP.items():
        for kw in keywords:
            if kw in msg_lower:
                return True, intent_type.value
    return False, ""


class TestIntentClassification:
    """意图分类评估（依赖 LLM API，仅测前 30 条控制成本）"""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not os.environ.get("DASHSCOPE_API_KEY") and not os.environ.get("DEEPSEEK_API_KEY"):
            pytest.skip("跳过意图分类评估: 需要配置 DASHSCOPE_API_KEY")

        self.router = IntentRouter()
        self.dataset = load_dataset("intent_200.json")

    @pytest.mark.asyncio
    async def test_all_cases_run(self):
        """验证测试数据集完整性"""
        assert len(self.dataset) >= 200

    @pytest.mark.asyncio
    async def test_intent_accuracy(self):
        """意图分类 Top-1 准确率（分 L2关键词 / L3 LLM兜底）"""
        # L2 统计：全部 200 条
        kw_total = 0
        kw_correct = 0
        for case in self.dataset:
            is_kw, kw_intent = _is_keyword_hit(case["message"])
            if is_kw:
                kw_total += 1
                if kw_intent == case["expected_intent"]:
                    kw_correct += 1

        # L3 统计：仅测前 30 条（控制 LLM API 成本）
        llm_total = 0
        llm_correct = 0
        all_correct = 0
        all_total = 0
        errors = []

        sample = self.dataset[:30]
        for case in sample:
            try:
                result = await self.router.classify(case["message"])
                predicted = result.intent.value
                expected = case["expected_intent"]
                is_kw, _ = _is_keyword_hit(case["message"])

                if not is_kw:
                    llm_total += 1
                    if predicted == expected:
                        llm_correct += 1

                if predicted == expected:
                    all_correct += 1
                else:
                    source = "关键词" if is_kw else "LLM"
                    errors.append(
                        f"  [{source}] '{case['message'][:35]}...' "
                        f"→ {predicted} (期望: {expected})"
                    )
                all_total += 1
            except Exception as e:
                errors.append(f"  ⚠ 异常: '{case['message'][:35]}...' → {e}")
                all_total += 1

        # 报告
        kw_acc = kw_correct / kw_total if kw_total > 0 else 1.0
        llm_acc = llm_correct / llm_total if llm_total > 0 else 1.0
        all_acc = all_correct / all_total if all_total > 0 else 0

        print(f"\n意图分类评估（全量 200 条 + 前 30 条 LLM）")
        print(f"  L2 关键词命中率: {kw_total}/{len(self.dataset)} ({kw_total/len(self.dataset)*100:.1f}%)")
        print(f"  L2 关键词准确率: {kw_correct}/{kw_total} = {kw_acc:.1%}")
        print(f"  L3 LLM 兜底样本: {llm_total}/{all_total}")
        print(f"  L3 LLM 准确率:    {llm_correct}/{llm_total} = {llm_acc:.1%}")
        print(f"  ──────────────────────────")
        print(f"  综合准确率(Top-1): {all_acc:.2%} ({all_correct}/{all_total})")
        for err in errors[:10]:
            print(err)

        assert kw_acc >= 0.80, f"L2 关键词准确率 {kw_acc:.1%} < 80%"
        assert all_acc >= 0.70, f"综合准确率 {all_acc:.2%} < 70%"

    @pytest.mark.asyncio
    async def test_metrics_report(self):
        """各类别准确率"""
        correct_by_intent = Counter()
        total_by_intent = Counter()

        for case in self.dataset[:30]:
            try:
                result = await self.router.classify(case["message"])
                expected = case["expected_intent"]
                total_by_intent[expected] += 1
                if result.intent.value == expected:
                    correct_by_intent[expected] += 1
            except Exception:
                pass

        print("\n各类别准确率（前30条）:")
        for intent, total in total_by_intent.most_common():
            correct = correct_by_intent.get(intent, 0)
            pct = correct / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 10)
            print(f"  {intent:12s}: {correct}/{total} ({pct:5.1f}%) {bar}")
