# -*- coding: utf-8 -*-
# EMR 适配器 - 模拟电子病历系统，提供患者就诊历史查询
from typing import Any, Dict, List, Optional


class EMRAdapter:
    """电子病历系统（EMR）模拟适配器，提供患者就诊记录查询"""

    # ---------- 模拟病历数据 ----------
    MOCK_RECORDS: List[Dict[str, Any]] = [
        {
            "record_id": "EMR202501001",
            "patient_name": "张三",
            "visit_date": "2025-01-15",
            "department": "内科",
            "chief_complaint": "发热、咳嗽3天",
            "diagnosis": "急性上呼吸道感染",
            "treatment": "阿莫西林胶囊 0.5g bid×3天，氨酚烷胺片 qd×3天",
            "doctor": "王医生",
            "follow_up": "3天后复查",
        },
        {
            "record_id": "EMR202501002",
            "patient_name": "张三",
            "visit_date": "2025-03-20",
            "department": "骨科",
            "chief_complaint": "右膝关节疼痛1周",
            "diagnosis": "右膝关节扭伤",
            "treatment": "布洛芬缓释胶囊 0.3g bid×5天，建议休息",
            "doctor": "赵医生",
            "follow_up": "1周后复查",
        },
        {
            "record_id": "EMR202501003",
            "patient_name": "张三",
            "visit_date": "2025-06-10",
            "department": "内分泌科",
            "chief_complaint": "口渴、多饮、多尿2月",
            "diagnosis": "2型糖尿病（初诊）",
            "treatment": "盐酸二甲双胍片 0.5g bid，饮食控制，适当运动",
            "doctor": "李主任",
            "follow_up": "1月后复查血糖",
        },
        {
            "record_id": "EMR202502001",
            "patient_name": "李四",
            "visit_date": "2025-02-10",
            "department": "消化内科",
            "chief_complaint": "上腹痛、反酸1月",
            "diagnosis": "胃食管反流病",
            "treatment": "奥美拉唑肠溶片 20mg qd×14天，蒙脱石散 3g tid",
            "doctor": "陈医生",
            "follow_up": "2周后复查",
        },
    ]

    @staticmethod
    async def get_patient_history(
        patient_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """根据患者姓名查询就诊历史记录"""
        records = [
            r for r in EMRAdapter.MOCK_RECORDS if r["patient_name"] == patient_name
        ]
        # 按就诊日期倒序排列
        records.sort(key=lambda x: x["visit_date"], reverse=True)
        return records[:limit]

    @staticmethod
    async def get_record_by_id(record_id: str) -> Optional[Dict[str, Any]]:
        """根据病历ID精确查询就诊记录"""
        for record in EMRAdapter.MOCK_RECORDS:
            if record["record_id"] == record_id:
                return record
        return None

    @staticmethod
    async def get_patient_summary(patient_name: str) -> Dict[str, Any]:
        """生成患者就诊摘要（总次数、最近就诊、诊断列表、用药列表）"""
        records = [
            r for r in EMRAdapter.MOCK_RECORDS if r["patient_name"] == patient_name
        ]
        if not records:
            return {
                "total_visits": 0,
                "last_visit": None,
                "diagnoses": [],
                "drugs_used": [],
            }

        # 按就诊日期排序
        records.sort(key=lambda x: x["visit_date"], reverse=True)
        last_visit = {
            "date": records[0]["visit_date"],
            "department": records[0]["department"],
            "diagnosis": records[0]["diagnosis"],
        }

        # 所有诊断列表（去重）
        diagnoses = list(set(r["diagnosis"] for r in records))

        # 用药列表（从treatment中提取药品名称）
        drugs_used = set()
        for r in records:
            treatment = r.get("treatment", "")
            # 简单提取：取空格前的药品名
            for part in treatment.split("，"):
                name = part.split(" ")[0].strip()
                if name and not name[0].isdigit():
                    drugs_used.add(name)

        return {
            "total_visits": len(records),
            "last_visit": last_visit,
            "diagnoses": diagnoses,
            "drugs_used": sorted(list(drugs_used)),
        }
