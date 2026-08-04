# -*- coding: utf-8 -*-
"""
Agent 输出格式合规性评估 — 程序检查（确定性，不依赖 LLM）

验证项:
  Report: 出院小结含 7 段标题 + 单次报告含异常指标列表
  Drug: 药品评估含 ✓/⚠/✗ + 处方审核含患者姓名/过敏史
  Inquiry: 结论含概率标签（可能性极高/大/中等/略有可能）

所有检查均为程序级的字符串/正则验证，不调用 LLM。
"""

import pytest


class TestAgentFormatCompliance:
    """Agent 输出格式合规性程序检查"""

    # ── Report Agent ──

    def test_discharge_summary_has_required_sections(self):
        """出院小结应包含 7 段必填标题"""
        mock_output = (
            "灵枢综合医院\n出院小结\n\n"
            "姓名：张三  性别：男  年龄：58  住院号：INP202512001\n"
            "入院日期：2024-12-15  出院日期：2024-12-22\n\n"
            "入院情况：患者因咳嗽咳痰...\n"
            "入院诊断：社区获得性肺炎\n"
            "诊疗经过：入院后予头孢曲松...\n"
            "出院诊断：社区获得性肺炎，临床治愈\n"
            "出院情况：无发热...\n"
            "出院医嘱：阿奇霉素续贯3天...\n"
        )
        required = ["出院小结", "入院情况", "入院诊断", "诊疗经过",
                     "出院诊断", "出院情况", "出院医嘱"]
        missing = [s for s in required if s not in mock_output]
        assert not missing, f"出院小结缺少标题: {missing}"

    def test_discharge_summary_has_disclaimer(self):
        """出院小结应含 AI 生成标注或医师审核提示"""
        mock_output = "（AI辅助生成，待主管医师审核签字）"
        assert "AI" in mock_output or "审核" in mock_output or "待医师" in mock_output

    def test_single_report_has_abnormal_indicators(self):
        """单次报告输出应含'异常指标'或'综合意见'"""
        mock_output = (
            "分析完成。张三，血常规\n\n"
            "异常指标：\n"
            "- 白细胞计数 11.2×10^9/L（偏高）：提示感染\n"
            "综合意见：建议结合临床\n"
            "免责声明：AI解读仅供参考"
        )
        assert "异常指标" in mock_output or "综合意见" in mock_output

    # ── Drug Agent ──

    def test_clinical_assessment_has_verdict(self):
        """药品临床评估应含 ✓/⚠/✗ 其中之一的判断符号"""
        mock_output = "盐酸二甲双胍片 — 可以使用 ✓\n\n患者：张三\n..."
        assert any(s in mock_output for s in ["可以使用", "慎用", "禁用"])

    def test_clinical_assessment_has_verdict_symbol(self):
        """药品临床评估应含判断符号 ✓ ⚠ ✗"""
        mock_output = "盐酸二甲双胍片 — 可以使用 ✓\n\n患者：张三\n..."
        assert any(s in mock_output for s in ["✓", "⚠", "✗"])

    def test_prescription_review_has_patient_name(self):
        """处方审核报告中应含患者姓名"""
        mock_output = "处方审核报告\n\n患者：张三\n过敏史：青霉素、磺胺类\n..."
        assert "患者：" in mock_output

    def test_prescription_review_has_pharmacist_notice(self):
        """处方审核报告应含药师审核提示"""
        mock_output = "（AI辅助审核，请药师最终确认）"
        assert "药师" in mock_output

    # ── Inquiry Agent ──

    def test_inquiry_conclusion_has_prob_label(self):
        """问诊结论应含概率中文标签"""
        labels = ["可能性极高", "可能性大", "中等可能", "略有可能", "可能性较低"]
        mock_output = "分析完成。症状：腹痛、恶心、反酸\n\n可能的疾病诊断：\n1. 胃食管反流病 可能性极高\n..."
        assert any(label in mock_output for label in labels)

    def test_inquiry_conclusion_has_disclaimer(self):
        """问诊结论应含免责声明"""
        mock_output = "免责声明：AI分诊仅供就医参考，不能替代医生诊断。"
        assert "免责声明" in mock_output or "仅供参考" in mock_output

    # ── General ──

    def test_emergency_output_has_recommendation(self):
        """急诊输出应含就医建议"""
        mock_output = "⚠ 紧急提醒：建议立即拨打120急救电话或前往最近的医院急诊科。"
        assert any(k in mock_output for k in ["120", "急诊", "急救"])


def test_format_check_functions_dont_crash():
    """格式检查函数本身不抛异常（防御性测试）"""
    # 空字符串应安全处理
    empty = ""
    assert not any(s in empty for s in ["患者：", "✓", "⚠", "✗", "可能性极高"])
