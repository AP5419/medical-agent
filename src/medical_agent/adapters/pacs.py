# -*- coding: utf-8 -*-
"""
Layer 6: 数据源层 - 数据适配器
模块: PACS/RIS（影像归档与通信系统/放射信息系统）模拟数据适配器
职责: 模拟影像检查记录的查询与影像报告数据提取
注意: 生产环境替换为真实 PACS/RIS 接口（DICOM / HL7 / FHIR / REST API）
"""

from typing import Optional

from loguru import logger

# ================================================================
# 模拟影像检查数据（PACS + RIS）
# 包含检查记录、DICOM 元数据、影像报告
# ================================================================

_MOCK_IMAGING_STUDIES = [
    {
        "study_id": "IMG20240701001",
        "patient_name": "张三",
        "patient_gender": "男",
        "patient_age": 45,
        "modality": "CT",
        "body_part": "胸部",
        "study_description": "胸部CT平扫",
        "study_date": "2024-07-01",
        "accession_number": "ACC20240701001",
        "referring_doctor": "刘医生",
        "department": "内科",
        "dicom_metadata": {
            "study_instance_uid": "1.2.840.113619.2.55.3.12345678.1.20240701.1",
            "slice_thickness": "1.25mm",
            "kvp": "120",
            "exposure": "200 mAs",
            "contrast_used": False,
        },
        "report": {
            "report_id": "RPT20240701001",
            "radiologist": "王医生",
            "report_date": "2024-07-01",
            "status": "已审核",
            "findings": "双肺纹理清晰，未见明显实质性病变。气管及主支气管通畅。纵隔淋巴结未见肿大。心脏大小形态正常。双侧胸腔未见积液。胸廓骨质未见明显异常。",
            "impression": "胸部CT平扫未见明显异常。",
            "recommendation": "建议结合临床随访。",
        },
    },
    {
        "study_id": "IMG20240701002",
        "patient_name": "李四",
        "patient_gender": "女",
        "patient_age": 58,
        "modality": "MRI",
        "body_part": "头部",
        "study_description": "头颅MRI平扫+增强",
        "study_date": "2024-07-01",
        "accession_number": "ACC20240701002",
        "referring_doctor": "陈医生",
        "department": "神经内科",
        "dicom_metadata": {
            "study_instance_uid": "1.2.840.113619.2.55.3.12345678.2.20240701.1",
            "slice_thickness": "5mm",
            "magnetic_field": "3.0T",
            "contrast_used": True,
            "contrast_agent": "钆喷酸葡胺 15ml",
        },
        "report": {
            "report_id": "RPT20240701002",
            "radiologist": "赵医生",
            "report_date": "2024-07-01",
            "status": "已审核",
            "findings": "右侧基底节区见斑片状长T1长T2信号影，DWI呈高信号，ADC图呈低信号，范围约1.5×1.2cm。增强扫描未见明显强化。脑室系统大小形态正常。中线结构居中。双侧大脑半球皮层下白质见散在斑点状长T2信号影。",
            "impression": "1. 右侧基底节区急性脑梗死；2. 双侧大脑半球皮层下白质缺血性改变。",
            "recommendation": "建议神经内科进一步治疗，定期复查。",
        },
    },
    {
        "study_id": "IMG20240701003",
        "patient_name": "王五",
        "patient_gender": "男",
        "patient_age": 52,
        "modality": "X线",
        "body_part": "胸部",
        "study_description": "胸部正位片",
        "study_date": "2024-07-01",
        "accession_number": "ACC20240701003",
        "referring_doctor": "李医生",
        "department": "心内科",
        "dicom_metadata": {
            "study_instance_uid": "1.2.840.113619.2.55.3.12345678.3.20240701.1",
            "kvp": "120",
            "exposure": "5 mAs",
        },
        "report": {
            "report_id": "RPT20240701003",
            "radiologist": "张医生",
            "report_date": "2024-07-01",
            "status": "已审核",
            "findings": "双肺纹理增粗，肺门影增大。心影呈靴型改变，心胸比约0.62。主动脉结增宽。双侧膈肌光滑，肋膈角锐利。",
            "impression": "1. 双肺纹理增粗；2. 心影增大，呈靴型心，符合高血压性心脏病改变。",
            "recommendation": "建议超声心动图进一步评估心功能。",
        },
    },
    {
        "study_id": "IMG20240620001",
        "patient_name": "张三",
        "patient_gender": "男",
        "patient_age": 45,
        "modality": "超声",
        "body_part": "腹部",
        "study_description": "腹部超声检查",
        "study_date": "2024-06-20",
        "accession_number": "ACC20240620001",
        "referring_doctor": "刘医生",
        "department": "消化内科",
        "dicom_metadata": {
            "study_instance_uid": "1.2.840.113619.2.55.3.12345678.4.20240620.1",
            "probe_type": "凸阵探头 3.5MHz",
        },
        "report": {
            "report_id": "RPT20240620001",
            "radiologist": "孙医生",
            "report_date": "2024-06-20",
            "status": "已审核",
            "findings": "肝脏大小形态正常，表面光滑，实质回声均匀。肝内管道结构走行正常。胆囊大小约7.2×2.8cm，壁光滑，腔内未见异常回声。胰腺大小形态正常，实质回声均匀。脾脏大小形态正常。双肾大小形态正常，实质回声正常，集合系统未见分离。",
            "impression": "腹部超声未见明显异常。",
            "recommendation": "暂无特殊处理。",
        },
    },
    {
        "study_id": "IMG20240615001",
        "patient_name": "李四",
        "patient_gender": "女",
        "patient_age": 58,
        "modality": "CT",
        "body_part": "腹部",
        "study_description": "腹部CT增强扫描",
        "study_date": "2024-06-15",
        "accession_number": "ACC20240615001",
        "referring_doctor": "王医生",
        "department": "内分泌科",
        "dicom_metadata": {
            "study_instance_uid": "1.2.840.113619.2.55.3.12345678.5.20240615.1",
            "slice_thickness": "5mm",
            "kvp": "120",
            "contrast_used": True,
            "contrast_agent": "碘海醇 100ml",
        },
        "report": {
            "report_id": "RPT20240615001",
            "radiologist": "周医生",
            "report_date": "2024-06-15",
            "status": "已审核",
            "findings": "肝脏密度弥漫性减低，肝/脾CT值比值约0.8（正常>1.0），符合中度脂肪肝表现。脾脏增大，约8个肋单元。胰腺形态正常，边缘光滑。双肾大小正常，肾实质密度均匀。腹腔及腹膜后未见明显肿大淋巴结。",
            "impression": "1. 中度脂肪肝；2. 脾脏轻度增大。",
            "recommendation": "建议控制饮食、增加运动，定期复查肝功能及肝脏超声。",
        },
    },
]

# 检查设备信息
_MOCK_EQUIPMENT = [
    {"ae_title": "CT-01", "device_name": "联影 uCT 960+", "modality": "CT", "location": "影像科CT室", "status": "运行中"},
    {"ae_title": "MRI-01", "device_name": "GE Signa Premier 3.0T", "modality": "MRI", "location": "影像科MRI室", "status": "运行中"},
    {"ae_title": "DR-01", "device_name": "西门子 Ysio Max", "modality": "X线", "location": "影像科DR室", "status": "运行中"},
    {"ae_title": "US-01", "device_name": "飞利浦 EPIQ 7", "modality": "超声", "location": "超声科1室", "status": "运行中"},
]


class PACSAdapter:
    """
    PACS/RIS（影像归档与通信系统 / 放射信息系统）模拟适配器

    职责:
        1. 查询患者影像检查记录
        2. 获取影像报告（放射科诊断结论）
        3. 按检查类型（CT/MRI/X线/超声）筛选
        4. 获取 DICOM 元数据

    PACS: 管理影像文件本身（DICOM 格式存储）
    RIS:  管理影像检查的工作流程和报告
    """

    def __init__(self):
        logger.info("PACS/RIS 适配器已初始化（模拟模式）")

    async def search_studies(
        self,
        patient_name: Optional[str] = None,
        modality: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        按患者姓名或检查类型查询影像检查记录

        Args:
            patient_name: 患者姓名（模糊匹配）
            modality: 检查类型（CT/MRI/X线/超声）
            limit: 返回记录上限
        """
        results = []
        for study in _MOCK_IMAGING_STUDIES:
            match = True
            if patient_name and patient_name not in study["patient_name"]:
                match = False
            if modality and modality not in study["modality"]:
                match = False
            if match:
                results.append(study)

        results.sort(key=lambda s: s["study_date"], reverse=True)
        logger.info(f"PACS 查询: patient={patient_name}, modality={modality} → {len(results[:limit])} 条")
        return results[:limit]

    async def get_study_by_id(self, study_id: str) -> Optional[dict]:
        """根据检查 ID 查询单条影像检查记录（含 DICOM 元数据）"""
        for study in _MOCK_IMAGING_STUDIES:
            if study["study_id"] == study_id:
                return study
        return None

    async def get_imaging_report(self, study_id: str) -> Optional[dict]:
        """
        获取影像报告的详细内容

        Returns:
            dict 包含: findings（影像所见）, impression（诊断意见）, recommendation（建议）
        """
        study = await self.get_study_by_id(study_id)
        if study and "report" in study:
            logger.info(f"PACS 影像报告: {study_id} → {study['report']['report_id']}")
            return study["report"]
        return None

    async def get_patient_imaging_summary(self, patient_name: str) -> dict:
        """
        获取患者的影像检查摘要（用于问诊 Agent 上下文注入）

        Returns:
            {
                "total_studies": int,
                "studies": [简单列表],
                "recent_abnormal": [异常发现列表]
            }
        """
        studies = await self.search_studies(patient_name)

        if not studies:
            return {"total_studies": 0, "studies": [], "recent_abnormal": []}

        # 提取最近异常的发现
        recent_abnormal = []
        for s in studies[:5]:
            report = s.get("report", {})
            impression = report.get("impression", "")
            findings = report.get("findings", "")
            # 检查是否包含异常关键词
            abnormal_keywords = ["异常", "病变", "结节", "阴影", "增大", "肿胀", "梗死", "出血", "炎症", "肿瘤", "增生"]
            is_abnormal = any(kw in impression or kw in findings for kw in abnormal_keywords)

            if is_abnormal:
                recent_abnormal.append({
                    "study_id": s["study_id"],
                    "date": s["study_date"],
                    "modality": s["modality"],
                    "body_part": s["body_part"],
                    "impression": impression,
                })

        return {
            "total_studies": len(studies),
            "studies": [
                {
                    "study_id": s["study_id"],
                    "date": s["study_date"],
                    "modality": s["modality"],
                    "body_part": s["body_part"],
                }
                for s in studies[:10]
            ],
            "recent_abnormal": recent_abnormal,
        }

    async def get_modality_statistics(self) -> list[dict]:
        """获取各检查类型的数量统计"""
        stats = {}
        for study in _MOCK_IMAGING_STUDIES:
            modality = study["modality"]
            stats[modality] = stats.get(modality, 0) + 1

        return [{"modality": k, "count": v} for k, v in stats.items()]

    async def get_equipment_list(self) -> list[dict]:
        """获取影像设备清单（AE Title + 设备型号）"""
        return _MOCK_EQUIPMENT
