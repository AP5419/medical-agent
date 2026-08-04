# -*- coding: utf-8 -*-
"""知识问答智能体 - 医学知识科普、临床指南查询、术语解释"""

import asyncio
from typing import Optional

from langchain.agents import create_agent
from langchain_core.tools import tool

from medical_agent.providers.llm import get_llm_qa


KNOWLEDGE_SYSTEM_PROMPT = """你是一位专业的医学知识问答助手。你提供循证医学知识、临床指南查询、医学术语解释等服务。

## 核心能力
1. **医学知识检索**：检索医学文献、教科书、临床指南中的相关知识
2. **知识图谱查询**：通过知识图谱查询疾病-症状-药品-检查之间的关联关系
3. **文档语料库搜索**：从药品说明书、临床指南、医学教育文档中搜索（非结构化文档）
4. **术语解释**：用通俗易懂的语言解释专业医学术语
5. **文档解析**：使用MinerU将PDF/DOCX/图片等医疗文档解析为结构化Markdown文本
6. **来源追溯**：所有答案必须标注可追溯的信息来源

## 工作流程
1. 接收用户医学知识问题
2. 同时调用文档检索、知识图谱查询和文档语料库搜索
3. 整合多源信息，形成综合答案
4. 对于需要通俗解释的内容，调用术语解释能力
5. 标注信息来源和置信度

## 回答要求
- **循证优先**：优先引用临床指南、系统评价等高质量证据
- **来源标注**：每条关键信息需标注来源出处
- **分层回答**：先给出简明结论，再展开详细解释
- **通俗易懂**：专业术语需附带通俗解释
- **区分确定性**：明确区分"已证实"、"一般共识"、"存在争议"的信息

## 免责声明
本知识问答仅供参考学习，不构成医疗建议。具体诊疗请咨询执业医师。
"""


@tool(response_format="content")
async def search_medical_docs(query: str) -> str:
    """搜索医学文档——从医学知识库中检索相关医学文献和临床指南

    Args:
        query: 医学知识查询内容
    """
    from medical_agent.engines.rag.medical_rag import MedicalRAGEngine
    engine = MedicalRAGEngine()
    docs = engine.search(query=query, top_k=5)
    if not docs:
        return "未找到相关医学文档。"
    import json
    return json.dumps(docs, ensure_ascii=False, indent=2)


@tool(response_format="content")
async def search_knowledge_graph(query: str) -> str:
    """搜索知识图谱——查询疾病-症状-药品-检查之间的关联关系

    Args:
        query: 知识图谱查询内容
    """
    from medical_agent.engines.graph.graph_rag import GraphRAGEngine
    engine = GraphRAGEngine()
    result = await engine.search(query=query)
    if isinstance(result, dict):
        return result.get("answer", str(result))
    return str(result)


@tool(response_format="content")
async def explain_medical_term(term: str) -> str:
    """解释医学术语——用通俗易懂的语言解释专业医学术语

    Args:
        term: 需要解释的医学术语
    """
    llm = get_llm_qa()
    explain_prompt = f"""请用通俗易懂的语言解释以下医学术语。要求：
1. 先给出简明定义（一句话）
2. 再展开详细解释
3. 举一个日常生活中的类比帮助理解
4. 如有相关注意事项请说明

医学术语：{term}

请用温暖、易懂的中文回答。"""

    result = await llm.ainvoke(explain_prompt)
    return result.content


@tool(response_format="content")
async def search_document_corpus(query: str) -> str:
    """搜索文档语料库——从药品说明书、临床指南、医学科普文档中搜索非结构化内容

    Args:
        query: 搜索查询内容
    """
    try:
        from medical_agent.engines.rag.graph_rag_ms import MSGraphRAGEngine
        engine = MSGraphRAGEngine()
        result = await engine.local_search(query)
        answer = result.get("answer", "")
        if not answer or "未初始化" in answer or "失败" in answer:
            return "文档语料库暂不可用（可能索引未构建或LLM未配置），建议使用其他工具。"
        return answer
    except Exception as e:
        return f"文档语料库搜索失败: {str(e)}。请使用其他工具获取信息。"


@tool(response_format="content")
async def parse_medical_document(file_path: str) -> str:
    """解析医疗文档——使用MinerU将PDF/DOCX/图片解析为结构化Markdown文本
    
    Args:
        file_path: 文档文件的路径（支持 PDF/DOCX/PPTX/XLSX/图片）
    """
    try:
        from medical_agent.engines.rag.mineru_client import MinerUClient
        client = MinerUClient()
        result = await client.parse_file(file_path)
        if result["success"]:
            return f"文档解析成功（{len(result['markdown'])}字符）：\n\n{result['markdown'][:3000]}"
        return f"文档解析失败：{result['error']}"
    except Exception as e:
        return f"文档解析异常：{str(e)}"


_lock = asyncio.Lock()
# 模块级知识问答智能体单例
_knowledge_agent: Optional[object] = None

async def get_knowledge_agent():
    global _knowledge_agent
    if _knowledge_agent is not None:
        return _knowledge_agent
    async with _lock:
        if _knowledge_agent is not None:
            return _knowledge_agent
        _knowledge_agent = create_knowledge_agent()
        return _knowledge_agent


def create_knowledge_agent():
    """创建知识问答智能体——含5个工具：文档检索、知识图谱查询、文档语料库搜索、术语解释、文档解析

    Returns:
        配置完成的知识问答智能体实例
    """
    global _knowledge_agent
    if _knowledge_agent is not None:
        return _knowledge_agent

    llm = get_llm_qa()
    tools = [search_medical_docs, search_knowledge_graph, search_document_corpus, explain_medical_term, parse_medical_document]

    _knowledge_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=KNOWLEDGE_SYSTEM_PROMPT,
    )

    return _knowledge_agent
