# -*- coding: utf-8 -*-
"""运营数据智能体 - 自然语言转SQL、管理统计数据查询、报表生成（管理员专用）"""

import asyncio
import contextvars
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from langchain.agents import create_agent
from langchain_core.tools import tool

from medical_agent.providers.llm import get_llm_qa


OPERATION_SYSTEM_PROMPT = """你是一个医疗运营数据查询助手。你的职责是将管理员的数据需求转化为SQL查询，生成统计报表。

## 核心能力
1. **自然语言转SQL**：将管理员的自然语言查询需求转换为SQL语句
2. **数据库模式理解**：了解数据库表结构和字段含义
3. **安全查询**：只执行聚合查询和统计分析，禁止访问个人敏感数据
4. **报表生成**：以表格或统计图表形式呈现查询结果

## 安全规则（必须严格遵守）
- **禁止查询个人数据**：不允许查询特定患者的个人信息、病历、处方
- **仅限聚合查询**：只允许执行COUNT、SUM、AVG、GROUP BY等聚合操作
- **禁止写操作**：不允许INSERT、UPDATE、DELETE、DROP等修改操作
- **数据脱敏**：查询结果中不得包含患者姓名、身份证号、手机号等敏感字段
- **权限校验**：仅管理员角色可以执行数据查询

## 支持的查询类型
1. **科室统计**：各科室挂号量、就诊量、收入统计
2. **药品统计**：药品使用排行、处方量统计
3. **运营指标**：日均门诊量、住院率、手术量
4. **时间趋势**：按日/周/月/年维度的指标趋势分析
5. **对比分析**：不同时间段、不同科室的对比

## 工作流程
1. 获取数据库模式信息（表结构）
2. 将用户自然语言查询转为SQL
3. 安全检查（过滤个人数据访问、确保聚合查询）
4. 执行SQL并返回格式化结果

## 注意事项
- 对于不明确的查询需求，先确认再执行
- 所有查询结果都是只读的
- 如查询涉及敏感数据，明确拒绝并说明原因
"""

# 数据库会话引用（运行时注入）
_db_session_ctx: contextvars.ContextVar = contextvars.ContextVar("db_session", default=None)


def set_db_session(session: AsyncSession) -> None:
    """运行时注入数据库会话

    Args:
        session: SQLAlchemy异步会话实例
    """
    _db_session_ctx.set(session)


@tool(response_format="content")
async def query_operation_data(query: str) -> str:
    """查询运营数据——将自然语言查询转为SQL并执行

    Args:
        query: 自然语言查询需求（如：查询本月各科室挂号量排行）
    """
    db_session = _db_session_ctx.get()
    if db_session is None:
        return "数据库会话未初始化，请联系管理员。"
    from medical_agent.engines.nl2sql.nl2sql import NL2SQLEngine
    import json
    engine = NL2SQLEngine(db_session=db_session)
    result = await engine.execute_query(question=query)
    return json.dumps(result, ensure_ascii=False, default=str)


@tool(response_format="content")
async def get_schema_info(table_name: Optional[str] = None) -> str:
    """获取数据库模式信息——返回表结构描述

    Args:
        table_name: 可选，指定表名；不指定则返回所有主要表结构
    """
    return """
数据库表结构:

- departments: 科室(id, name, description)
  科室ID映射: 1=内科, 2=外科, 3=妇产科, 4=儿科, 5=五官科, 6=皮肤科, 7=眼科

- diseases: 疾病(id, name, department_id, description, cause, prevent, cure_way, cure_lasttime, cured_prob, cost_money, easy_get)

- symptoms: 症状(id, name)

- disease_symptoms: 疾病-症状关联(id, disease_id, symptom_id)

- drugs: 药品(id, name, alias, category, manufacturer, approval_number, is_otc, stock_quantity, price, expire_date)

- drug_details: 药品详情(id, drug_id, indication, usage_dosage, adverse_reaction, contraindication, precaution, interaction, storage, full_instruction)

- disease_drugs: 疾病-药品关联(id, disease_id, drug_id, relation_type)

- patients: 患者(id, name, gender, age, phone, id_card, blood_type, allergy_history, medical_history)

- consultations: 问诊(id, patient_id, department_id, chief_complaint, diagnosis, prescription, urgency_level, created_at)

- users: 系统用户(id, username, role, phone, email, is_active)
"""


_lock = asyncio.Lock()
# 模块级运营数据智能体单例
_operation_agent: Optional[object] = None

async def get_operation_agent():
    global _operation_agent
    if _operation_agent is not None:
        return _operation_agent
    async with _lock:
        if _operation_agent is not None:
            return _operation_agent
        _operation_agent = create_operation_agent()
        return _operation_agent


def create_operation_agent():
    """创建运营数据智能体——含2个工具：数据查询、模式信息

    Returns:
        配置完成的运营数据智能体实例
    """
    global _operation_agent
    if _operation_agent is not None:
        return _operation_agent

    llm = get_llm_qa()
    tools = [query_operation_data, get_schema_info]

    _operation_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=OPERATION_SYSTEM_PROMPT,
    )

    return _operation_agent
