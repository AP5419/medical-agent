# -*- coding: utf-8 -*-
# HIS 适配器 - 模拟医院信息系统，提供药品查询、处方管理等接口
from typing import Any, Dict, List, Optional


class HISAdapter:
    """医院信息系统（HIS）模拟适配器，提供药品和处方数据"""

    # ---------- 模拟药品数据 ----------
    # 药品编码、名称、规格、库存、价格、厂家
    MOCK_DRUGS: List[Dict[str, Any]] = [
        {
            "drug_code": "DRUG001",
            "name": "阿莫西林胶囊",
            "aliases": ["阿莫西林"],
            "spec": "0.5g×24粒",
            "stock": 500,
            "price": 12.50,
            "manufacturer": "华北制药",
        },
        {
            "drug_code": "DRUG002",
            "name": "布洛芬缓释胶囊",
            "aliases": ["布洛芬"],
            "spec": "0.3g×20粒",
            "stock": 300,
            "price": 18.00,
            "manufacturer": "中美史克",
        },
        {
            "drug_code": "DRUG003",
            "name": "奥美拉唑肠溶片",
            "aliases": ["奥美拉唑"],
            "spec": "20mg×14片",
            "stock": 200,
            "price": 35.00,
            "manufacturer": "阿斯利康",
        },
        {
            "drug_code": "DRUG004",
            "name": "盐酸二甲双胍片",
            "aliases": ["二甲双胍", "格华止"],
            "spec": "0.5g×30片",
            "stock": 400,
            "price": 22.00,
            "manufacturer": "中美施贵宝",
        },
        {
            "drug_code": "DRUG005",
            "name": "硝苯地平控释片",
            "aliases": ["硝苯地平", "拜新同"],
            "spec": "30mg×7片",
            "stock": 150,
            "price": 28.50,
            "manufacturer": "拜耳医药",
        },
        {
            "drug_code": "DRUG006",
            "name": "氯雷他定片",
            "aliases": ["氯雷他定", "开瑞坦"],
            "spec": "10mg×6片",
            "stock": 350,
            "price": 15.00,
            "manufacturer": "先声药业",
        },
        {
            "drug_code": "DRUG007",
            "name": "蒙脱石散",
            "aliases": ["蒙脱石", "思密达"],
            "spec": "3g×10袋",
            "stock": 250,
            "price": 20.00,
            "manufacturer": "博福益普生",
        },
        {
            "drug_code": "DRUG008",
            "name": "复方氨酚烷胺片",
            "aliases": ["复方氨酚烷胺", "快克", "感康"],
            "spec": "12片",
            "stock": 600,
            "price": 9.80,
            "manufacturer": "仁和药业",
        },
    ]

    # ---------- 模拟处方数据 ----------
    MOCK_PRESCRIPTIONS: List[Dict[str, Any]] = [
        {
            "prescription_id": "P20250101",
            "patient_name": "张三",
            "patient_id": "P202501001",
            "drugs": [
                {"drug_code": "DRUG004", "name": "盐酸二甲双胍片", "quantity": 1, "dose": "500mg bid"},
                {"drug_code": "DRUG005", "name": "硝苯地平控释片", "quantity": 1, "dose": "30mg qd"},
            ],
            "department": "内分泌科",
            "status": "待审核",
        },
        {
            "prescription_id": "P20250102",
            "patient_name": "李四",
            "patient_id": "P20250210001",
            "drugs": [
                {"drug_code": "DRUG003", "name": "奥美拉唑肠溶片", "quantity": 1, "dose": "20mg qd"},
                {"drug_code": "DRUG007", "name": "蒙脱石散", "quantity": 1, "dose": "3g tid"},
            ],
            "department": "消化内科",
            "status": "已取药",
        },
        {
            "prescription_id": "P20250103",
            "patient_name": "张三",
            "patient_id": "P202501001",
            "drugs": [
                {"drug_code": "DRUG001", "name": "阿莫西林胶囊", "quantity": 2, "dose": "0.5g tid"},
            ],
            "department": "呼吸内科",
            "status": "已取药",
        },
    ]

    # ---------- 患者信息（过敏史、基础病史） ----------
    PATIENT_INFO: Dict[str, Dict[str, Any]] = {
        "P202501001": {
            "patient_name": "张三",
            "patient_id": "P202501001",
            "allergies": ["青霉素", "磺胺类"],
            "conditions": ["2型糖尿病", "高血压1级", "高血压"],
        },
        "P20250210001": {
            "patient_name": "李四",
            "patient_id": "P20250210001",
            "allergies": [],
            "conditions": ["慢性胃炎"],
        },
    }

    @staticmethod
    async def search_drugs(keyword: str) -> List[Dict[str, Any]]:
        """根据关键字模糊搜索药品（名称、代码、别名）"""
        keyword_lower = keyword.lower()
        results = []
        for drug in HISAdapter.MOCK_DRUGS:
            if keyword_lower in drug["name"].lower() or keyword_lower in drug["drug_code"].lower():
                results.append(drug)
                continue
            for alias in drug.get("aliases", []):
                if keyword_lower in alias.lower():
                    results.append(drug)
                    break
        return results

    @staticmethod
    async def get_drug_by_code(code: str) -> Optional[Dict[str, Any]]:
        """根据药品编码精确查询药品"""
        for drug in HISAdapter.MOCK_DRUGS:
            if drug["drug_code"] == code:
                return drug
        return None

    @staticmethod
    async def check_stock(code: str) -> Optional[Dict[str, Any]]:
        """检查药品库存（返回库存信息）"""
        drug = await HISAdapter.get_drug_by_code(code)
        if drug is None:
            return None
        return {
            "drug_code": drug["drug_code"],
            "name": drug["name"],
            "stock": drug["stock"],
            "available": drug["stock"] > 0,
        }

    @staticmethod
    async def get_prescriptions(patient_name: str) -> List[Dict[str, Any]]:
        """根据患者姓名查询处方记录"""
        return [
            p
            for p in HISAdapter.MOCK_PRESCRIPTIONS
            if p["patient_name"] == patient_name
        ]

    @staticmethod
    async def get_patient_info(patient_name: str) -> Optional[Dict[str, Any]]:
        """获取患者完整信息（处方 + 过敏史 + 病史）"""
        prescriptions = await HISAdapter.get_prescriptions(patient_name)
        info = None
        for pid, pinfo in HISAdapter.PATIENT_INFO.items():
            if pinfo.get("patient_name") == patient_name:
                info = pinfo
                break
        return {
            "patient_name": patient_name,
            "allergies": info.get("allergies", []) if info else [],
            "conditions": info.get("conditions", []) if info else [],
            "prescriptions": prescriptions,
        }

    @staticmethod
    async def submit_prescription(
        patient_name: str, drugs: List[Dict[str, Any]], department: str
    ) -> Dict[str, Any]:
        """提交新处方（模拟）"""
        prescription_id = f"P{len(HISAdapter.MOCK_PRESCRIPTIONS) + 1:06d}"
        new_prescription = {
            "prescription_id": prescription_id,
            "patient_name": patient_name,
            "drugs": drugs,
            "department": department,
            "status": "待审核",
        }
        HISAdapter.MOCK_PRESCRIPTIONS.append(new_prescription)
        return new_prescription
