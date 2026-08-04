# -*- coding: utf-8 -*-
"""GraphRAG引擎 - 基于Neo4j知识图谱的医疗关系查询"""

import json
import re
from typing import Optional

from loguru import logger

from medical_agent.core.config import get_settings
from medical_agent.providers.llm import get_llm_qa


class GraphRAGEngine:
    """医疗知识图谱RAG引擎，支持NL2Cypher、多跳推理、药物相互作用检测"""

    # 知识图谱Schema描述，用于NL2Cypher提示词
    GRAPH_SCHEMA = """
知识图谱包含以下节点类型和关系：

节点类型：
- Disease（疾病）：name(名称), category(类别), description(描述)
- Symptom（症状）：name(名称), description(描述)
- Drug（药品）：name(名称), category(类别), usage(用法), dosage(剂量), side_effects(副作用)
- Department（科室）：name(名称), description(描述)
- Check（检查项目）：name(名称), description(描述), preparation(准备事项)
- Food（食物）：name(名称), category(类别)

关系类型：
- HAS_SYMPTOM：(Disease)-[HAS_SYMPTOM]->(Symptom) 疾病包含的症状
- BELONGS_TO：(Disease)-[BELONGS_TO]->(Department) 疾病所属科室
- COMMON_DRUG：(Disease)-[COMMON_DRUG]->(Drug) 疾病常用药品
- RECOMMEND_DRUG：(Disease)-[RECOMMEND_DRUG]->(Drug) 疾病推荐药品
- NEED_CHECK：(Disease)-[NEED_CHECK]->(Check) 疾病需要的检查
- ACOMPANY_WITH：(Disease)-[ACOMPANY_WITH]->(Disease) 疾病伴随关系
- DO_EAT：(Drug)-[DO_EAT]->(Food) 服药期间适宜食物
- NO_EAT：(Drug)-[NO_EAT]->(Food) 服药期间禁忌食物
"""

    # 合法标签白名单（从 GRAPH_SCHEMA 提取）
    _VALID_LABELS = {"Disease", "Symptom", "Drug", "Department", "Check", "Food"}

    # 合法关系白名单
    _VALID_RELATIONS = {
        "HAS_SYMPTOM", "BELONGS_TO", "COMMON_DRUG", "RECOMMEND_DRUG",
        "NEED_CHECK", "ACOMPANY_WITH", "DO_EAT", "NO_EAT",
    }

    def __init__(self):
        """初始化GraphRAG引擎，获取Neo4j驱动和LLM实例"""
        from medical_agent.infra.neo4j import get_neo4j_driver

        self.driver = get_neo4j_driver()
        self.llm = get_llm_qa()
        self.settings = get_settings()

    # ---------- Cypher 验证工具 ----------

    @staticmethod
    def _extract_cypher_labels(cypher: str) -> set:
        """从 Cypher 语句中提取所有节点标签（`:Label` 模式）

        Args:
            cypher: Cypher 查询语句

        Returns:
            标签名集合（大写）
        """
        # 匹配 :Label 模式。排除变量名中的冒号（如 :param）
        labels = re.findall(r':([A-Za-z_]+)', cypher)
        return {lbl.upper() for lbl in labels}

    @staticmethod
    def _extract_cypher_relations(cypher: str) -> set:
        """从 Cypher 语句中提取所有关系类型（`[:REL_TYPE]` 模式）

        Args:
            cypher: Cypher 查询语句

        Returns:
            关系名集合（大写）
        """
        # 匹配 [:REL_TYPE] 或 [:REL_TYPE|REL_TYPE2] 模式
        rel_matches = re.findall(r'\[:([A-Za-z_|]+)\]', cypher)
        relations = set()
        for match in rel_matches:
            for rel in match.replace(" ", "").split("|"):
                relations.add(rel.upper())
        return relations

    def _validate_cypher_schema(self, cypher: str) -> tuple[bool, str]:
        """白名单校验：检查 Cypher 中引用的标签和关系是否在合法列表中

        EXPlAIN 只校验语法，不校验标签/关系是否存在。
        这一层确保 LLM 不会编造不存在的标签名。

        Args:
            cypher: Cypher 查询语句

        Returns:
            (is_valid, error_message)
        """
        labels = self._extract_cypher_labels(cypher)
        relations = self._extract_cypher_relations(cypher)

        # 校验标签
        unknown_labels = labels - self._VALID_LABELS
        if unknown_labels:
            return False, f"使用了不在白名单中的标签: {unknown_labels}。合法标签: {self._VALID_LABELS}"

        # 校验关系
        unknown_relations = relations - self._VALID_RELATIONS
        if unknown_relations:
            return False, f"使用了不在白名单中的关系: {unknown_relations}。合法关系: {self._VALID_RELATIONS}"

        return True, ""

    async def _validate_cypher_semantics(self, user_query: str, cypher: str) -> tuple[bool, str]:
        """语义回译校验：将 Cypher 翻译回自然语言，检查是否符合用户意图

        这是 soft-warning 机制——不匹配时记录日志但不拒绝查询，
        因为 LLM-as-Judge 本身也可能出错，需要积累数据后再设硬阈值。

        Args:
            user_query: 用户原始查询
            cypher: 生成的 Cypher

        Returns:
            (matches_intent, reasoning)
        """
        try:
            # Step 1: Cypher → 自然语言解释
            explain_prompt = (
                f"请用一句中文简要解释以下 Cypher 查询做了什么：\n{cypher}\n\n解释："
            )
            explain_resp = self.llm.invoke(explain_prompt)
            explanation = explain_resp.content.strip() if hasattr(explain_resp, "content") else str(explain_resp).strip()

            # Step 2: 判断解释是否匹配用户意图
            judge_prompt = (
                f"用户问题是：\"{user_query}\"\n"
                f"数据库查询的解释是：\"{explanation}\"\n\n"
                f"请判断这条查询是否能回答用户的问题。返回JSON: "
                f'{{"match": true/false, "reason": "简要说明"}}'
            )
            judge_resp = self.llm.invoke(judge_prompt)
            judge_text = judge_resp.content.strip() if hasattr(judge_resp, "content") else str(judge_resp).strip()

            # 解析 JSON
            try:
                # 移除可能的代码块标记
                if judge_text.startswith("```"):
                    lines = judge_text.split("\n")
                    judge_text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
                result = json.loads(judge_text)
                matches = result.get("match", True)
                reason = result.get("reason", "")
            except json.JSONDecodeError:
                # 回退：检查文本中是否包含否定词
                matches = "false" not in judge_text.lower()
                reason = judge_text[:200]

            return matches, reason

        except Exception as e:
            logger.warning(f"语义回译校验异常: {e}")
            return True, "校验异常，默认放行"

    # ---------- 实体抽取 ----------
        """从查询中提取医学实体

        Args:
            query: 用户自然语言查询

        Returns:
            包含 diseases, symptoms, drugs, departments 键的字典
        """
        prompt = f"""你是一个医学实体抽取助手。请从以下查询中提取医学实体，以JSON格式返回。
只返回JSON，不要额外解释。

字段说明：
- diseases: 疾病名称列表
- symptoms: 症状名称列表
- drugs: 药品名称列表
- departments: 科室名称列表

查询：{query}

JSON输出："""
        response = self.llm.invoke(prompt)
        text = response.content.strip() if hasattr(response, "content") else str(response).strip()

        # 清理可能的markdown代码块标记
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "diseases": [],
                "symptoms": [],
                "drugs": [],
                "departments": [],
            }

    async def nl_to_cypher(self, query: str, max_retries: int = 2) -> Optional[str]:
        """将自然语言查询转换为Cypher查询语句，带多层验证

        验证链（5层，按执行顺序）：
        ① Prompt 约束 — 注入 Schema 和规则
        ② 格式检查 — 非 MATCH 开头 → 拒绝
        ③ 标签/关系白名单 — 不在合法列表 → 重试
        ④ EXPLAIN 语法验证 — Neo4j 预编译
        ⑤ 语义回译校验 — Cypher→NL 解释是否符合意图（soft warning）

        Args:
            query: 用户自然语言查询
            max_retries: 最大重试次数

        Returns:
            Cypher查询语句，失败返回None
        """
        base_prompt = f"""你是一个Neo4j Cypher查询专家。请根据以下知识图谱Schema和用户查询，生成Cypher查询语句。
只返回Cypher语句本身，不要任何解释或代码块标记。

{self.GRAPH_SCHEMA}

重要规则：
- 使用MATCH进行查询，只返回SELECT类操作
- 关系名称全部大写并用方括号括起来
- 节点标签仅限: {', '.join(sorted(self._VALID_LABELS))}
- 关系类型仅限: {', '.join(sorted(self._VALID_RELATIONS))}
- 如果查询可能不准确，使用CONTAINS或正则表达式进行模糊匹配
- 只生成只读查询（MATCH/RETURN），禁止任何写入操作
- 使用LIMIT控制结果数量

用户查询：{query}

Cypher："""

        prompt = base_prompt

        for attempt in range(max_retries + 1):
            try:
                response = self.llm.invoke(prompt)
                cypher = response.content.strip() if hasattr(response, "content") else str(response).strip()

                # 清理代码块标记
                if cypher.startswith("```"):
                    lines = cypher.split("\n")
                    cypher = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
                cypher = cypher.strip()

                # ──────────── Layer ②: 格式检查 ────────────
                if not cypher.upper().startswith("MATCH"):
                    if attempt < max_retries:
                        prompt = base_prompt + "\n\n上次生成的语句不符合要求（不是MATCH开头），请严格只生成MATCH查询语句。"
                        continue
                    return None

                # ──────────── Layer ③: 标签/关系白名单校验 ────────────
                valid, err_msg = self._validate_cypher_schema(cypher)
                if not valid:
                    logger.warning(f"白名单校验失败: {err_msg}")
                    if attempt < max_retries:
                        prompt = base_prompt + f"\n\n上次生成的语句使用了不合法的标签或关系：{err_msg}"
                        continue
                    return None

                # ──────────── Layer ④: EXPLAIN 语法验证 ────────────
                try:
                    async with self.driver.session() as session:
                        await session.run(f"EXPLAIN {cypher}")
                except Exception as e:
                    if attempt < max_retries:
                        prompt = base_prompt + f"\n\n上次生成的Cypher语法有误：{str(e)}\n请修正后重新生成。"
                        continue
                    return None

                # ──────────── Layer ⑤: 语义回译校验 (soft-warning) ────────────
                matches, reason = await self._validate_cypher_semantics(query, cypher)
                if not matches:
                    logger.warning(f"语义回译校验不匹配: query='{query[:50]}...', cypher='{cypher[:80]}...', reason='{reason}'")
                    # soft-warning: 记录日志但不拒绝。当数据显示误判率高时再设硬阈值
                    # TODO: 积累100+样本后，分析回译校验的准确率，决定是否升级为硬拦截

                return cypher

            except Exception as e:
                if attempt < max_retries:
                    prompt = base_prompt + f"\n\n生成失败：{str(e)}，请重新生成。"
                    continue
                return None

        return None

    async def execute_cypher(self, cypher: str, parameters: dict = None) -> list[dict]:
        """执行Cypher查询并返回结果记录

        Args:
            cypher: Cypher查询语句
            parameters: 查询参数绑定

        Returns:
            查询结果列表，每条记录为字典
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(cypher, parameters or {})
                records = []
                for record in result:
                    row = {}
                    for key, value in record.items():
                        if hasattr(value, "_properties"):
                            row[key] = dict(value._properties)
                            if hasattr(value, "labels"):
                                row[key]["_labels"] = list(value.labels)
                        else:
                            row[key] = value
                    records.append(row)
                return records
        except Exception as e:
            return [{"error": str(e)}]

    async def search(self, query: str) -> dict:
        """完整的知识图谱搜索流程：实体抽取 → NL2Cypher → 执行 → 生成答案

        Args:
            query: 用户自然语言查询

        Returns:
            包含 answer, entities, graph_data, cypher 的字典
        """
        # 抽取实体
        entities = self.extract_entities(query)

        # 转换Cypher
        cypher = await self.nl_to_cypher(query)

        # 执行查询
        graph_data = []
        if cypher:
            graph_data = await self.execute_cypher(cypher)

        # 生成答案
        answer = self._generate_graph_answer(query, graph_data, cypher)

        return {
            "answer": answer,
            "entities": entities,
            "graph_data": graph_data,
            "cypher": cypher,
        }

    async def multi_hop_search(self, start_entity: str, hops: int = 2) -> list[dict]:
        """多跳路径遍历，探索实体之间的关系链

        Args:
            start_entity: 起始实体名称
            hops: 跳数（路径深度）

        Returns:
            多跳路径结果列表
        """
        if hops < 1:
            hops = 2

        cypher = """
MATCH path = (n)-[*1..$hops]-(connected)
WHERE n.name CONTAINS $start_entity
RETURN n.name AS start_node,
       connected.name AS end_node,
       length(path) AS distance,
       [rel IN relationships(path) | type(rel)] AS relation_types,
       [node IN nodes(path) | labels(node)[0]] AS node_labels
LIMIT 30
"""
        return await self.execute_cypher(cypher, parameters={"start_entity": start_entity, "hops": hops})

    async def check_drug_interaction(self, drug_a: str, drug_b: str) -> Optional[dict]:
        """检查两种药物之间是否存在已知的相互作用（通过共享疾病）

        Args:
            drug_a: 药品A名称
            drug_b: 药品B名称

        Returns:
            相互作用信息字典，无相互作用返回None
        """
        cypher = """
MATCH (d1:Drug {name: $drug_a})<-[:COMMON_DRUG|RECOMMEND_DRUG]-(disease:Disease)-[:COMMON_DRUG|RECOMMEND_DRUG]->(d2:Drug {name: $drug_b})
RETURN disease.name AS shared_disease,
       d1.name AS drug_a,
       d2.name AS drug_b,
       disease.description AS disease_description
LIMIT 10
"""
        results = await self.execute_cypher(cypher, parameters={"drug_a": drug_a, "drug_b": drug_b})
        if results and "error" not in results[0]:
            return {
                "has_interaction": True,
                "drug_a": drug_a,
                "drug_b": drug_b,
                "shared_diseases": results,
                "warning": f"发现 {drug_a} 和 {drug_b} 共同关联以下疾病，可能存在联合用药风险。"
            }
        return None

    def _generate_graph_answer(
        self, query: str, graph_data: list[dict], cypher: Optional[str]
    ) -> str:
        """使用LLM对图谱查询结果进行总结归纳"""
        if not graph_data:
            return "未在知识图谱中找到相关信息。"

        if len(graph_data) > 0 and "error" in graph_data[0]:
            return f"图谱查询出错：{graph_data[0]['error']}"

        # 序列化图谱数据为文本
        graph_text = json.dumps(graph_data, ensure_ascii=False, indent=2)
        if len(graph_text) > 4000:
            graph_text = graph_text[:4000] + "\n...(结果已截断)"

        prompt = f"""你是一个医学知识图谱分析助手。请根据以下知识图谱查询结果，用通俗易懂的语言回答用户问题。

用户问题：{query}

图谱查询结果：
{graph_text}

请用中文总结归纳要点，帮助用户理解。如果信息不完整，请如实说明。"""
        response = self.llm.invoke(prompt)
        return response.content.strip() if hasattr(response, "content") else str(response).strip()
