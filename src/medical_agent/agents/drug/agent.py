# -*- coding: utf-8 -*-
"""药物咨询智能体 - 药品信息查询、药物相互作用检查、处方安全审核"""

import asyncio
from typing import Optional

from langchain.agents import create_agent
from langchain_core.tools import tool

from medical_agent.providers.llm import get_llm_qa


DRUG_SYSTEM_PROMPT = """你是一位专业的药物咨询助手。你能提供药品信息查询、药物相互作用检查、处方安全审核等服务。

## 服务等级
### 浅层服务（面向患者/医生基础查询）
- 药品说明书查询（适应症、用法用量、不良反应、禁忌症）
- 药品基本信息（通用名、商品名、剂型、规格）
- 同类药品比较
- 用药注意事项

### 深层服务（面向药师处方审核）
- **多跳推理**：基于知识图谱的跨实体多步推理，分析药品→靶点→通路→疾病→副作用链路
- **药物相互作用检查**：检查多种药物之间的相互作用（CYP450酶、血浆蛋白结合、协同/拮抗）
- **处方安全审核**：审核处方的合理性（剂量、禁忌人群、配伍禁忌）
- **用药方案优化**：基于患者特征（年龄、肝肾功能、基因型）推荐个体化用药方案

## 工作流程
1. 接收用户药物咨询问题
2. 确定服务等级（浅层/深层）
3. 查询药品信息（HIS药品数据库）
4. 如涉及多种药物或多跳推理，调用药物相互作用检查
5. 对于深度处方审核需求，执行多跳图推理
6. 综合评估并生成建议

## 注意事项
- 明确区分浅层和深层服务
- 药物相互作用检查应覆盖药代动力学和药效学两个维度
- 处方审核时应标注风险等级
- 所有用药建议均需附免责声明

## 免责声明
本药物咨询仅供参考，不构成处方建议。请严格遵医嘱用药，切勿自行调整用药方案。
"""


@tool(response_format="content")
async def search_drug_info(drug_name: str) -> str:
    """搜索药品信息——从HIS药品数据库获取药品详细信息

    Args:
        drug_name: 药品名称（通用名或商品名）
    """
    from medical_agent.adapters.his import HISAdapter
    import json
    adapter = HISAdapter()
    drugs = await adapter.search_drugs(keyword=drug_name)
    if not drugs:
        return f"未找到药品「{drug_name}」的相关信息。"
    drug_info_str = json.dumps(drugs[0], ensure_ascii=False) if len(drugs) == 1 else json.dumps(drugs, ensure_ascii=False)
    return drug_info_str


@tool(response_format="content")
async def check_drug_interaction(drug_names: str) -> str:
    """检查药物相互作用——基于知识图谱分析多药物之间的潜在相互作用

    Args:
        drug_names: 药品名称列表，多个用逗号分隔（如：阿司匹林,华法林,布洛芬）
    """
    from medical_agent.engines.graph.graph_rag import GraphRAGEngine
    import json
    engine = GraphRAGEngine()
    drug_list = [d.strip() for d in drug_names.split(",") if d.strip()]
    if len(drug_list) < 2:
        return "需要至少两种药品才能检查相互作用。"

    interactions = []
    for i in range(len(drug_list)):
        for j in range(i+1, len(drug_list)):
            result = engine.check_drug_interaction(drug_list[i], drug_list[j])
            if result:
                interactions.append(json.dumps(result, ensure_ascii=False, default=str))
    result_str = "; ".join(interactions) if interactions else "未发现已知药物相互作用。"
    return result_str


@tool(response_format="content")
async def multi_hop_drug_reasoning(query: str) -> str:
    """多跳药物推理——基于知识图谱进行药品→靶点→通路→疾病→副作用的跨实体多步推理

    Args:
        query: 药物多跳推理查询（如：分析二甲双胍的作用机制和潜在副作用链路）
    """
    from medical_agent.engines.graph.graph_rag import GraphRAGEngine
    import json
    engine = GraphRAGEngine()
    result = engine.multi_hop_search(start_entity=query, hops=4)
    return json.dumps(result, ensure_ascii=False, default=str)


@tool(response_format="content")
async def review_prescription_safety(prescription_text: str) -> str:
    """审核处方安全性——审核处方的合理性，包括剂量、禁忌、配伍、人群适用性

    Args:
        prescription_text: 处方内容文本
    """
    llm = get_llm_qa()
    review_prompt = f"""请审核以下处方的安全性，检查以下方面：
1. 药品剂量是否在合理范围
2. 是否存在配伍禁忌
3. 是否考虑特殊人群（老人/孕妇/儿童/肝肾不全）
4. 是否存在重复用药
5. 给药途径是否合理

处方内容：
{prescription_text}

请按风险等级（高/中/低）分类列出审核发现，并给出建议。"""

    result = await llm.ainvoke(review_prompt)
    return result.content


_lock = asyncio.Lock()
# 模块级药物咨询智能体单例
_drug_agent: Optional[object] = None

async def get_drug_agent():
    global _drug_agent
    if _drug_agent is not None:
        return _drug_agent
    async with _lock:
        if _drug_agent is not None:
            return _drug_agent
        _drug_agent = create_drug_agent()
        return _drug_agent


def create_drug_agent():
    """创建药物咨询智能体——含4个工具：药品搜索、相互作用检查、多跳推理、处方审核

    Returns:
        配置完成的药物咨询智能体实例
    """
    global _drug_agent
    if _drug_agent is not None:
        return _drug_agent

    llm = get_llm_qa()
    tools = [
        search_drug_info,
        check_drug_interaction,
        multi_hop_drug_reasoning,
        review_prescription_safety,
    ]

    _drug_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=DRUG_SYSTEM_PROMPT,
    )

    return _drug_agent
