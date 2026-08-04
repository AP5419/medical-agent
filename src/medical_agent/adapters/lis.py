# -*- coding: utf-8 -*-
# LIS 适配器 - 模拟实验室信息系统，提供检验报告查询和指标解读
from typing import Any, Dict, List, Optional


class LISAdapter:
    """实验室信息系统（LIS）模拟适配器，提供检验报告和指标知识"""

    # ---------- 模拟检验报告数据（张三，社区获得性肺炎，4时间点） ----------
    MOCK_REPORTS: List[Dict[str, Any]] = [
        # ── 12/15 入院 ──
        {
            "id": "L20251215001", "patient_name": "张三", "patient_id": "P202501001",
            "report_type": "血常规", "report_date": "2024-12-15",
            "indicators": [
                {"name": "白细胞计数", "value": 15.3, "unit": "×10^9/L", "ref_range": "3.5-9.5", "status": "偏高"},
                {"name": "中性粒细胞百分比", "value": 85.0, "unit": "%", "ref_range": "40-75", "status": "偏高"},
                {"name": "超敏C反应蛋白", "value": 85.0, "unit": "mg/L", "ref_range": "<5", "status": "偏高"},
                {"name": "降钙素原", "value": 2.5, "unit": "ng/mL", "ref_range": "<0.5", "status": "偏高"},
                {"name": "血小板计数", "value": 310, "unit": "×10^9/L", "ref_range": "125-350", "status": "正常"},
            ],
        },
        {
            "id": "L20251215002", "patient_name": "张三", "patient_id": "P202501001",
            "report_type": "生化检查", "report_date": "2024-12-15",
            "indicators": [
                {"name": "谷丙转氨酶", "value": 38, "unit": "U/L", "ref_range": "9-50", "status": "正常"},
                {"name": "肌酐", "value": 82, "unit": "μmol/L", "ref_range": "44-133", "status": "正常"},
                {"name": "血糖", "value": 6.1, "unit": "mmol/L", "ref_range": "3.9-6.1", "status": "正常"},
            ],
        },
        # ── 12/18 D3 治疗后 ──
        {
            "id": "L20251218001", "patient_name": "张三", "patient_id": "P202501001",
            "report_type": "血常规", "report_date": "2024-12-18",
            "indicators": [
                {"name": "白细胞计数", "value": 12.1, "unit": "×10^9/L", "ref_range": "3.5-9.5", "status": "偏高"},
                {"name": "中性粒细胞百分比", "value": 78.0, "unit": "%", "ref_range": "40-75", "status": "偏高"},
                {"name": "超敏C反应蛋白", "value": 42.0, "unit": "mg/L", "ref_range": "<5", "status": "偏高"},
                {"name": "降钙素原", "value": 0.8, "unit": "ng/mL", "ref_range": "<0.5", "status": "偏高"},
            ],
        },
        # ── 12/20 D5 ──
        {
            "id": "L20251220001", "patient_name": "张三", "patient_id": "P202501001",
            "report_type": "血常规", "report_date": "2024-12-20",
            "indicators": [
                {"name": "白细胞计数", "value": 9.8, "unit": "×10^9/L", "ref_range": "3.5-9.5", "status": "正常"},
                {"name": "中性粒细胞百分比", "value": 68.0, "unit": "%", "ref_range": "40-75", "status": "正常"},
                {"name": "超敏C反应蛋白", "value": 22.0, "unit": "mg/L", "ref_range": "<5", "status": "偏高"},
                {"name": "降钙素原", "value": 0.3, "unit": "ng/mL", "ref_range": "<0.5", "status": "正常"},
            ],
        },
        # ── 12/22 出院 ──
        {
            "id": "L20251222001", "patient_name": "张三", "patient_id": "P202501001",
            "report_type": "血常规", "report_date": "2024-12-22",
            "indicators": [
                {"name": "白细胞计数", "value": 7.2, "unit": "×10^9/L", "ref_range": "3.5-9.5", "status": "正常"},
                {"name": "中性粒细胞百分比", "value": 58.0, "unit": "%", "ref_range": "40-75", "status": "正常"},
                {"name": "超敏C反应蛋白", "value": 6.0, "unit": "mg/L", "ref_range": "<5", "status": "正常"},
                {"name": "降钙素原", "value": 0.1, "unit": "ng/mL", "ref_range": "<0.5", "status": "正常"},
            ],
        },
        # 保留李四——验证多患者查询
        {
            "id": "L20250210001", "patient_name": "李四", "patient_id": "P20250210001",
            "report_type": "尿常规", "report_date": "2025-02-10",
            "indicators": [
                {"name": "尿蛋白", "value": "+", "unit": "", "ref_range": "阴性", "status": "异常"},
                {"name": "尿白细胞", "value": "++", "unit": "", "ref_range": "阴性", "status": "异常"},
                {"name": "pH值", "value": 6.0, "unit": "", "ref_range": "5.0-7.0", "status": "正常"},
            ],
        },
    ]

    # ---------- 患者入院信息（出院小结用） ----------
    PATIENT_ADMISSION: Dict[str, Dict[str, Any]] = {
        "P202501001": {
            "patient_name": "张三",
            "patient_id": "P202501001",
            "gender": "男",
            "age": 58,
            "hospital_name": "灵枢综合医院",
            "admission_number": "INP202512001",
            "admission_date": "2024-12-15",
            "discharge_date": "2024-12-22",
            "chief_complaint": "咳嗽咳痰3天，发热1天，伴右侧胸痛",
            "admission_diagnosis": "社区获得性肺炎（右下叶）",
            "discharge_diagnosis": "社区获得性肺炎（右下叶），临床治愈",
            "vital_signs": "T 38.5℃, P 96/min, R 22/min, BP 128/76mmHg",
            "treatment": "头孢曲松 2g qd ivgtt + 阿奇霉素 0.5g qd po (12/15-12/22)",
            "CT_summary": "12/15 CT: 右下肺后基底段大片状实变，空气支气管征(+)。12/20 CT: 实变较前吸收约70%。",
            "discharge_orders": "阿奇霉素0.5g qd po 续贯3天；1周后门诊复查血常规+CRP；避免劳累，注意休息。",
            "discharge_condition": "出院时无发热，无咳嗽咳痰，无胸痛。查体：T 36.5℃，双肺呼吸音清，未闻及干湿啰音。",
        }
    }

    # ---------- 检验指标知识库 ----------
    INDICATOR_KNOWLEDGE: Dict[str, str] = {
        "白细胞计数": "白细胞是免疫系统的核心细胞，升高提示感染或炎症，降低可能与病毒感染、药物或骨髓抑制有关。",
        "红细胞计数": "红细胞负责运输氧气，减少提示贫血，增多可能与脱水或真性红细胞增多症有关。",
        "血红蛋白": "血红蛋白是红细胞中的携氧蛋白，降低提示贫血，增高见于脱水或高原适应。",
        "血小板计数": "血小板参与凝血过程，减少有出血风险，增多有血栓风险。",
        "中性粒细胞百分比": "中性粒细胞是白细胞的主要成分，升高提示细菌感染或急性炎症。",
        "超敏C反应蛋白": "CRP是急性时相反应蛋白，显著升高提示细菌感染或严重炎症，动态监测可评估抗感染治疗效果。",
        "降钙素原": "PCT是细菌感染的特异性标志物，>0.5ng/mL提示细菌感染可能，动态下降提示治疗有效。",
        "谷丙转氨酶": "ALT是肝细胞损伤的标志物，升高提示肝细胞损伤，常见于肝炎、脂肪肝等。",
        "谷草转氨酶": "AST存在于肝脏、心肌等组织，升高提示肝损伤或心肌损伤。",
        "总胆红素": "胆红素是血红蛋白代谢产物，升高可致黄疸，提示肝病或胆道梗阻。",
        "肌酐": "肌酐是肌肉代谢废物，升高提示肾功能减退。",
        "血糖": "血糖是血液中的葡萄糖，升高常见于糖尿病，降低可能导致低血糖昏迷。",
        "尿蛋白": "尿中出现蛋白提示肾脏滤过功能受损，常见于肾炎、肾病综合征等。",
        "尿糖": "尿糖阳性提示血糖过高超出肾糖阈，常见于糖尿病。",
        "尿白细胞": "尿中白细胞增多提示泌尿系统感染，如尿道炎、膀胱炎。",
        "尿红细胞": "尿中红细胞提示泌尿系统出血，需排查结石、感染或肿瘤。",
        "pH值": "尿液酸碱度，偏酸可能与饮食或代谢性酸中毒有关，偏碱可能与感染或素食有关。",
    }

    @staticmethod
    async def search_reports(
        patient_name: str, report_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """根据患者姓名和报告类型查询检验报告"""
        results = []
        for report in LISAdapter.MOCK_REPORTS:
            if report["patient_name"] == patient_name:
                if report_type is None or report_type in ("all", "") or report["report_type"] == report_type:
                    results.append(report)
        return results

    @staticmethod
    async def get_report_by_id(report_id: str) -> Optional[Dict[str, Any]]:
        """根据报告ID精确查询检验报告"""
        for report in LISAdapter.MOCK_REPORTS:
            if report["id"] == report_id:
                return report
        return None

    @staticmethod
    async def get_abnormal_indicators(report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提取检验报告中的异常指标"""
        abnormal = []
        for indicator in report.get("indicators", []):
            if indicator.get("status") not in ("正常",):
                abnormal.append(indicator)
        return abnormal

    @staticmethod
    async def get_indicator_knowledge(indicator_name: str) -> Optional[str]:
        """根据指标名称查询临床意义知识"""
        return LISAdapter.INDICATOR_KNOWLEDGE.get(indicator_name)

    @staticmethod
    async def get_patient_lis_report(patient_id: str = "", patient_name: str = "") -> Dict[str, Any]:
        """获取患者全部 LIS 检验报告（按时间排序）"""
        reports = []
        for r in LISAdapter.MOCK_REPORTS:
            if (patient_name and r.get("patient_name") == patient_name) or \
               (patient_id and r.get("patient_id") == patient_id):
                reports.append(r)
        reports.sort(key=lambda x: x.get("report_date", ""))
        return {"reports": reports, "count": len(reports)}

    @staticmethod
    async def get_admission_info(patient_id: str = "", patient_name: str = "") -> Optional[Dict[str, Any]]:
        """获取患者入院信息（出院小结用）"""
        for pid, info in LISAdapter.PATIENT_ADMISSION.items():
            if (patient_id and pid == patient_id) or \
               (patient_name and info.get("patient_name") == patient_name):
                return info
        return None
