# -*- coding: utf-8 -*-
# 意图路由器测试 - 仅测试 detect_emergency 方法（无 LLM 依赖）
import pytest

from medical_agent.orchestration.intent_router import IntentRouter


class TestIntentRouter:
    """意图路由器测试"""

    @pytest.fixture
    def router(self):
        """创建 IntentRouter 实例（不加载 LLM）"""
        return IntentRouter()

    def test_detect_emergency_returns_tuple(self, router):
        """测试 detect_emergency 返回 tuple[bool, list]"""
        result = router.detect_emergency("我有点头痛")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], list)

    def test_detect_emergency_chest_pain(self, router):
        """测试检测胸痛为紧急情况"""
        is_emergency, keywords = router.detect_emergency("我突然胸痛得厉害")
        assert is_emergency is True
        assert "胸痛" in keywords

    def test_detect_emergency_breathing_difficulty(self, router):
        """测试检测呼吸困难为紧急情况"""
        is_emergency, keywords = router.detect_emergency("病人出现呼吸困难，需要急救")
        assert is_emergency is True
        assert "呼吸困难" in keywords

    def test_detect_emergency_unconscious(self, router):
        """测试检测意识不清为紧急情况"""
        is_emergency, keywords = router.detect_emergency("患者意识不清，请立即处理")
        assert is_emergency is True
        assert "意识不清" in keywords

    def test_detect_emergency_coma(self, router):
        """测试检测昏迷为紧急情况"""
        is_emergency, keywords = router.detect_emergency("有人昏迷了")
        assert is_emergency is True

    def test_detect_emergency_stroke(self, router):
        """测试检测中风为紧急情况"""
        is_emergency, keywords = router.detect_emergency("怀疑是中风，半边身体动不了")
        assert is_emergency is True
        assert "中风" in keywords

    def test_detect_emergency_shock(self, router):
        """测试检测休克为紧急情况"""
        is_emergency, keywords = router.detect_emergency("病人休克了")
        assert is_emergency is True
        assert "休克" in keywords

    def test_detect_non_emergency_mild_symptom(self, router):
        """测试轻微症状不是紧急情况"""
        is_emergency, keywords = router.detect_emergency("我最近有点咳嗽，不严重")
        assert is_emergency is False
        assert keywords == []

    def test_detect_non_emergency_greeting(self, router):
        """测试问候语不是紧急情况"""
        is_emergency, keywords = router.detect_emergency("你好，我想咨询个问题")
        assert is_emergency is False
        assert keywords == []

    def test_detect_non_emergency_drug_inquiry(self, router):
        """测试药品咨询不是紧急情况"""
        is_emergency, keywords = router.detect_emergency("阿莫西林一天吃几次？")
        assert is_emergency is False
        assert keywords == []

    def test_detect_emergency_empty_message(self, router):
        """测试空消息不是紧急情况"""
        is_emergency, keywords = router.detect_emergency("")
        assert is_emergency is False
        assert keywords == []

    def test_detect_emergency_multiple_keywords(self, router):
        """测试包含多个紧急关键词时全部检出"""
        is_emergency, keywords = router.detect_emergency("患者出现胸痛和呼吸困难，可能是心肌梗死")
        assert is_emergency is True
        assert "胸痛" in keywords
        assert "呼吸困难" in keywords
        assert "心肌梗死" in keywords

    def test_detect_emergency_case_insensitive(self, router):
        """测试关键词大小写不敏感（中文测试通过中文关键词）"""
        is_emergency, keywords = router.detect_emergency("有严重的吐血症状")
        assert is_emergency is True
        assert "吐血" in keywords
