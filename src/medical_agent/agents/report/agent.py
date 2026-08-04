# -*- coding: utf-8 -*-
"""报告解读智能体 - 化验单/影像报告/体检报告解读"""

import asyncio
from typing import Optional

from langchain.agents import create_agent
from langchain_core.tools import tool

from medical_agent.providers.llm import get_llm_qa


REPORT_SYSTEM_PROMPT = """你是一位专业的医学报告解读助手。你的职责是解读各种医学报告，包括化验单、影像报告、体检报告等。

## 核心能力
1. **报告分析**：识别报告中的异常指标，标注超出参考范围的项目
2. **指标解读**：用通俗易懂的语言解释每项异常指标的含义
3. **影像报告查询**：查询PACS/RIS系统中的影像检查记录和放射科诊断报告（CT/MRI/X线/超声）
4. **风险评估**：基于异常指标组合，评估可能的健康风险
5. **就诊建议**：给出初步的就诊方向和复查建议

## 工作流程
1. 接收报告内容或报告图片
2. 分析报告中的关键指标（含影像分析——VLM）
3. 识别异常项目并计算偏离程度
4. 如有需要，调用PACS查询患者历史影像报告
5. 查询医学知识库获取指标临床意义
6. 生成结构化的报告解读结果

## 解读报告时
- 逐项列出异常指标及其临床意义
- 说明异常程度（轻度/中度/重度偏离）
- 结合多项指标进行综合评估（化验+影像联合分析）
- 给出进一步检查或就诊建议

## 注意事项
- 对所有异常指标都用通俗语言解释
- 标注哪些指标需要紧急关注
- 解读结论应按紧急程度排序
- 避免引起不必要的恐慌，语气温和专业

## 免责声明
本报告解读仅供参考，不能替代执业医师的诊断。如有异常指标，请及时就医，由专业医生进行确诊。
"""


@tool(response_format="content")
async def analyze_report_image(image_base64: str) -> str:
    """分析报告图片——使用VLM（视觉语言模型）识别报告内容

    Args:
        image_base64: 报告的Base64编码图片
    """
    from medical_agent.engines.vlm.vlm_client import VLMClient
    import tempfile, os, base64
    client = VLMClient()
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp.write(base64.b64decode(image_base64))
        tmp_path = tmp.name
    result = client.analyze_medical_image(tmp_path, "请识别并提取这份医学报告中的所有关键信息和指标数值。")
    os.unlink(tmp_path)
    return result


@tool(response_format="content")
async def search_lab_reports(patient_id: str, report_type: str = "all") -> str:
    """搜索化验报告——从LIS系统获取患者的检验报告数据

    Args:
        patient_id: 患者ID
        report_type: 报告类型（血常规/生化检查/尿常规，查全部填 all）
    """
    from medical_agent.adapters.lis import LISAdapter
    adapter = LISAdapter()
    reports = await adapter.search_reports(patient_name=patient_id, report_type=report_type)
    if not reports:
        return "未找到该患者的检验报告。"
    return str(reports)


@tool(response_format="content")
async def search_imaging_studies(patient_name: str, modality: str = "all") -> str:
    """搜索影像检查——从PACS/RIS系统获取患者的影像检查记录和放射科诊断报告

    Args:
        patient_name: 患者姓名
        modality: 检查类型（CT/MRI/X线/超声/all）
    """
    from medical_agent.adapters.pacs import PACSAdapter
    import json
    adapter = PACSAdapter()
    modality_filter = None if modality == "all" else modality
    studies = await adapter.search_studies(patient_name=patient_name, modality=modality_filter)

    if not studies:
        return "未找到该患者的影像检查记录。"

    # 同时获取每个检查的影像报告
    result_parts = []
    for s in studies[:5]:
        report = s.get("report", {})
        result_parts.append({
            "study_id": s["study_id"],
            "date": s["study_date"],
            "modality": s["modality"],
            "body_part": s["body_part"],
            "description": s["study_description"],
            "findings": report.get("findings", ""),
            "impression": report.get("impression", ""),
            "radiologist": report.get("radiologist", ""),
            "status": report.get("status", ""),
        })

    return json.dumps(result_parts, ensure_ascii=False, indent=2)


@tool(response_format="content")
async def get_indicator_knowledge(indicator_name: str) -> str:
    """获取指标知识——查询医学检验指标的正常范围、临床意义

    Args:
        indicator_name: 检验指标名称（如：白细胞计数、谷丙转氨酶）
    """
    from medical_agent.engines.rag.medical_rag import MedicalRAGEngine
    engine = MedicalRAGEngine()
    result = engine.search(
        f"{indicator_name} 正常参考范围 临床意义 异常原因",
        top_k=3,
    )
    return str(result)


_lock = asyncio.Lock()
# 模块级报告智能体单例
_report_agent: Optional[object] = None

async def get_report_agent():
    global _report_agent
    if _report_agent is not None:
        return _report_agent
    async with _lock:
        if _report_agent is not None:
            return _report_agent
        _report_agent = create_report_agent()
        return _report_agent


def create_report_agent():
    """创建报告解读智能体——含4个工具：图片分析、化验搜索、影像查询、指标知识

    Returns:
        配置完成的报告解读智能体实例
    """
    global _report_agent
    if _report_agent is not None:
        return _report_agent

    llm = get_llm_qa()
    tools = [analyze_report_image, search_lab_reports, search_imaging_studies, get_indicator_knowledge]

    _report_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=REPORT_SYSTEM_PROMPT,
    )

    return _report_agent
