# -*- coding: utf-8 -*-
"""NL2SQL引擎 - 自然语言转SQL查询，支持安全校验、脱敏处理"""

import re
import asyncio
from typing import Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from medical_agent.providers.llm import get_llm_qa


class NL2SQLEngine:
    """自然语言转SQL引擎，包含SQL生成、安全校验、结果脱敏等完整链路"""

    # MySQL数据库Schema描述，包含中文注释
    DB_SCHEMA = """
数据库 schema:

表: departments (科室)
  - id: INT, 主键
  - name: VARCHAR(100), 科室名称
  - description: TEXT, 科室简介

表: diseases (疾病)
  - id: INT, 主键
  - name: VARCHAR(200), 疾病名称
  - department_id: INT, 外键关联 departments.id
  - description: TEXT, 疾病描述
  - cause: TEXT, 病因
  - prevent: TEXT, 预防措施
  - cure_way: TEXT, 治疗方式
  - cure_lasttime: VARCHAR(200), 治愈周期
  - cured_prob: VARCHAR(200), 治愈概率
  - cost_money: VARCHAR(200), 治疗费用
  - easy_get: TEXT, 易感人群

表: symptoms (症状)
  - id: INT, 主键
  - name: VARCHAR(200), 症状名称

表: disease_symptoms (疾病-症状关联)
  - id: INT, 主键
  - disease_id: INT, 外键关联 diseases.id
  - symptom_id: INT, 外键关联 symptoms.id

表: drugs (药品)
  - id: INT, 主键
  - name: VARCHAR(200), 药品名称
  - alias: VARCHAR(200), 药品别名
  - category: VARCHAR(200), 药品类别
  - manufacturer: VARCHAR(200), 生产厂家
  - approval_number: VARCHAR(200), 批准文号
  - is_otc: BOOLEAN, 是否OTC
  - stock_quantity: INT, 库存数量
  - price: FLOAT, 单价
  - expire_date: VARCHAR(50), 有效期

表: drug_details (药品详情)
  - id: INT, 主键
  - drug_id: INT, 外键关联 drugs.id, 唯一
  - indication: TEXT, 适应症
  - usage_dosage: TEXT, 用法用量
  - adverse_reaction: TEXT, 不良反应
  - contraindication: TEXT, 禁忌
  - precaution: TEXT, 注意事项
  - interaction: TEXT, 药物相互作用
  - storage: TEXT, 贮藏
  - full_instruction: TEXT, 完整说明书

表: disease_drugs (疾病-药品关联)
  - id: INT, 主键
  - disease_id: INT, 外键关联 diseases.id
  - drug_id: INT, 外键关联 drugs.id
  - relation_type: VARCHAR(20), 关联类型(common/recommend)

表: patients (患者)
  - id: INT, 主键
  - name: VARCHAR(100), 姓名
  - gender: VARCHAR(10), 性别
  - age: INT, 年龄
  - phone: VARCHAR(50), 电话
  - id_card: VARCHAR(50), 身份证号
  - allergy_history: TEXT, 过敏史
  - medical_history: TEXT, 既往病史
  - blood_type: VARCHAR(10), 血型

表: consultations (问诊记录)
  - id: INT, 主键
  - patient_id: INT, 外键关联 patients.id
  - department_id: INT, 外键关联 departments.id
  - chief_complaint: TEXT, 主诉
  - diagnosis: TEXT, 诊断结论
  - prescription: TEXT, 处方
  - urgency_level: VARCHAR(20), 紧急程度
  - created_at: DATETIME, 创建时间

表: users (系统用户)
  - id: INT, 主键
  - username: VARCHAR(50), 用户名
  - role: VARCHAR(20), 角色 (patient/doctor/admin)
  - is_active: BOOLEAN, 是否启用
"""

    # 禁止的SQL模式
    FORBIDDEN_PATTERNS = [
        r"\bDROP\b",
        r"\bDELETE\b",
        r"\bINSERT\b",
        r"\bUPDATE\b",
        r"\bALTER\b",
        r"\bCREATE\b",
        r"\bTRUNCATE\b",
        r"\bEXEC\b",
        r"\bEXECUTE\b",
        r"\-\-",
        r"\/\*",
        r"INTO\s+OUTFILE",
        r"LOAD_FILE",
    ]

    # 敏感字段列表
    SENSITIVE_FIELDS = ["id_card", "password_hash", "phone", "email"]

    def __init__(self, db_session: AsyncSession):
        """初始化NL2SQL引擎

        Args:
            db_session: SQLAlchemy异步数据库会话
        """
        self.db_session = db_session
        self.llm = get_llm_qa()

    def generate_sql(self, question: str, max_retries: int = 2) -> Optional[str]:
        """使用LLM生成MySQL SELECT SQL语句

        Args:
            question: 用户自然语言问题
            max_retries: 最大重试次数

        Returns:
            生成的SQL语句，失败返回None
        """
        prompt = f"""你是一个MySQL 8.0 SQL专家。请根据以下数据库Schema和用户问题，生成合法的MySQL SELECT查询语句。
只返回SQL语句本身，不要任何解释、代码块标记或分号后的内容。

{self.DB_SCHEMA}

重要规则：
1. 只生成SELECT查询语句，禁止任何INSERT/UPDATE/DELETE/ALTER/CREATE/DROP等操作
2. 禁止查询患者的个人隐私信息（手机号、邮箱、身份证号、密码哈希等）
3. 表名和字段名使用反引号包裹
4. 字符串值使用单引号包裹
5. 使用LIMIT控制结果数量，不超过100条
6. 使用JOIN时需要明确条件
7. 如果问题涉及多个表，优先使用LEFT JOIN

用户问题：{question}

SQL："""

        for attempt in range(max_retries + 1):
            try:
                response = self.llm.invoke(prompt)
                sql = response.content.strip() if hasattr(response, "content") else str(response).strip()

                # 清理代码块标记
                if sql.startswith("```"):
                    lines = sql.split("\n")
                    sql = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
                sql = sql.strip()

                # 清理末尾分号
                if sql.endswith(";"):
                    sql = sql[:-1].strip()

                if not sql.upper().startswith("SELECT"):
                    if attempt < max_retries:
                        prompt = f"{prompt}\n\n上次生成的语句不符合要求（不是SELECT开头），请严格只生成SELECT查询语句。"
                        continue
                    return None

                # 使用EXPLAIN验证语法
                valid, error_msg = self.validate_sql(sql)
                if not valid:
                    if attempt < max_retries:
                        prompt = f"{prompt}\n\n上次生成的SQL校验失败：{error_msg}\n请修正后重新生成。"
                        continue
                    return None

                return sql

            except Exception as e:
                if attempt < max_retries:
                    prompt = f"{prompt}\n\n生成失败：{str(e)}，请重新生成。"
                    continue
                return None

        return None

    @staticmethod
    def validate_sql(sql: str) -> Tuple[bool, str]:
        """校验SQL语句的安全性

        Args:
            sql: SQL语句

        Returns:
            (是否通过, 错误信息)
        """
        if not sql:
            return False, "SQL语句为空"

        sql_upper = sql.upper().strip()

        # 检查是否以SELECT开头
        if not sql_upper.startswith("SELECT"):
            return False, "SQL语句不是SELECT查询"

        # 检查禁止模式
        for pattern in NL2SQLEngine.FORBIDDEN_PATTERNS:
            if re.search(pattern, sql, re.IGNORECASE):
                return False, f"SQL包含禁止的操作模式: {pattern}"

        # 检查敏感字段
        sql_lower = sql.lower()
        for field in NL2SQLEngine.SENSITIVE_FIELDS:
            if field in sql_lower:
                return False, f"SQL包含敏感字段: {field}"

        return True, ""

    @staticmethod
    def desensitize_results(results: list) -> list:
        """对查询结果中的敏感字段进行脱敏处理

        Args:
            results: 查询结果列表

        Returns:
            脱敏后的结果列表
        """
        desensitized = []
        for row in results:
            new_row = dict(row._mapping) if hasattr(row, "_mapping") else dict(row)
            for key, value in new_row.items():
                if value is None:
                    continue
                key_lower = key.lower()
                if "phone" in key_lower or "mobile" in key_lower:
                    value_str = str(value)
                    if len(value_str) >= 7:
                        new_row[key] = value_str[:3] + "****" + value_str[-4:]
                elif "id_card" in key_lower or "idcard" in key_lower:
                    value_str = str(value)
                    if len(value_str) >= 18:
                        new_row[key] = value_str[:3] + "***********" + value_str[-4:]
                    elif len(value_str) > 6:
                        new_row[key] = value_str[:3] + "****" + value_str[-2:]
                elif "password" in key_lower:
                    new_row[key] = "******"
                elif "email" in key_lower:
                    value_str = str(value)
                    if "@" in value_str:
                        local, domain = value_str.split("@", 1)
                        if len(local) > 2:
                            local = local[:2] + "***"
                        new_row[key] = f"{local}@{domain}"
            desensitized.append(new_row)
        return desensitized

    async def execute_query(self, question: str) -> dict:
        """完整的NL2SQL查询流程：生成SQL → 校验 → 执行 → 脱敏 → 生成答案

        Args:
            question: 用户自然语言问题

        Returns:
            包含 success, sql, results, total, answer, error 的字典
        """
        # 步骤1：生成SQL
        sql = self.generate_sql(question)
        if not sql:
            return {
                "success": False,
                "sql": None,
                "results": [],
                "total": 0,
                "answer": "无法生成有效的SQL查询语句。",
                "error": "SQL生成失败",
            }

        # 步骤2：二次校验
        valid, error_msg = self.validate_sql(sql)
        if not valid:
            return {
                "success": False,
                "sql": sql,
                "results": [],
                "total": 0,
                "answer": f"SQL安全校验未通过：{error_msg}",
                "error": error_msg,
            }

        # 步骤3：执行SQL（10秒超时）
        try:
            result = await asyncio.wait_for(
                self.db_session.execute(text(sql)),
                timeout=10.0,
            )
            rows = result.fetchall()
        except asyncio.TimeoutError:
            return {
                "success": False,
                "sql": sql,
                "results": [],
                "total": 0,
                "answer": "SQL执行超时（超过10秒），请简化查询条件。",
                "error": "SQL执行超时",
            }
        except Exception as e:
            return {
                "success": False,
                "sql": sql,
                "results": [],
                "total": 0,
                "answer": f"SQL执行失败：{str(e)}",
                "error": str(e),
            }

        # 步骤4：数据脱敏
        desensitized = self.desensitize_results(rows)

        # 步骤5：生成自然语言答案
        answer = self._generate_answer(question, sql, desensitized)

        return {
            "success": True,
            "sql": sql,
            "results": desensitized,
            "total": len(desensitized),
            "answer": answer,
            "error": None,
        }

    def _generate_answer(
        self, question: str, sql: str, results: list
    ) -> str:
        """使用LLM将查询结果总结为自然语言答案"""
        if not results:
            return "查询未返回任何数据，请检查查询条件。"

        # 截断结果避免过长
        results_text = str(results)
        if len(results_text) > 3000:
            results_text = results_text[:3000] + "\n...(结果已截断)"

        prompt = f"""你是一个数据库查询分析助手。请用自然语言中文回答用户的问题。

用户问题：{question}

执行的SQL：{sql}

查询结果（已脱敏处理）：
{results_text}

请用简洁清晰的中文总结查询结果，如果数据较多请概括性地描述。"""
        response = self.llm.invoke(prompt)
        return response.content.strip() if hasattr(response, "content") else str(response).strip()
