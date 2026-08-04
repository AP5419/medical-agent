# -*- coding: utf-8 -*-
"""
症状标准化模块 — 三层匹配管道

① LLM 提取 + 标准化：口语 → 医学术语（如"拉肚子"→"腹泻"）
② Neo4j 精确匹配：验证术语是否在知识图谱中存在
③ Milvus 语义兜底：向量相似度匹配未命中的词汇

"""

import json
import re
from loguru import logger

# ── 第一层：LLM 提取 Prompt ────────────────────────────────────────────────

SYMPTOM_EXTRACT_PROMPT = """你是医疗术语标准化专家。

任务：从用户的描述中提取所有症状，并将每个症状转换为标准医学术语。

标准化规则（不限于此，尽量标准化）：
- 发烧/烧/低烧/高烧 → 发热
- 肚子疼/肚痛/腹部疼痛/肚子不舒服 → 腹痛
- 头晕/头晕眼花/天旋地转 → 眩晕
- 喘不上气/憋气/气短/胸闷喘气 → 呼吸困难
- 拉肚子/跑肚/稀便/大便不成形 → 腹泻
- 心跳快/心慌/心跳加速/心跳不规律 → 心悸
- 浑身没劲/没力气/疲惫/全身乏力 → 乏力
- 嗓子疼/喉咙疼/咽喉痛 → 咽痛
- 胸口疼/胸部疼痛/前胸痛 → 胸痛
- 恶心想吐/想呕吐/胃部不适 → 恶心
- 头疼/头部疼痛/偏头痛 → 头痛
- 流鼻涕/鼻涕/鼻塞流涕 → 流涕
- 咳嗽/干咳/咳痰 → 咳嗽
- 失眠/睡不着/入睡困难/易醒 → 失眠
- 便秘/大便干结/排便困难 → 便秘

用户描述：{user_input}

请返回JSON格式：{{"symptoms": ["标准化症状1", "标准化症状2"]}}
如果没有明确症状，返回 {{"symptoms": []}}。"""

# Milvus 语义相似度阈值（余弦相似度）
SIMILARITY_THRESHOLD = 0.85


async def extract_symptoms_layer1(user_input: str, llm) -> list[str]:
    """
    第一层：LLM 提取 + 标准化

    Args:
        user_input: 用户原始描述
        llm: LLM 实例（from medical_agent.providers.llm.get_llm_qa）

    Returns:
        标准化症状名列表，失败返回空列表
    """
    prompt = SYMPTOM_EXTRACT_PROMPT.format(user_input=user_input)
    try:
        response = llm.invoke(prompt)
        text = response.content.strip() if hasattr(response, "content") else str(response).strip()

        # 清理代码块标记
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        # 提取 JSON
        json_match = re.search(r'\{[^{}]*\}', text)
        if json_match:
            data = json.loads(json_match.group())
            symptoms = [s.strip() for s in data.get("symptoms", []) if s.strip()]
            logger.debug(f"LLM 提取症状: {symptoms}")
            return symptoms
    except Exception as e:
        logger.warning(f"LLM 症状提取失败: {e}")

    return []


async def match_neo4j_layer2(symptoms: list[str], neo4j_driver) -> tuple[list[str], list[str]]:
    """
    第二层：Neo4j 精确匹配

    Args:
        symptoms: 标准化症状名列表
        neo4j_driver: Neo4j 异步驱动

    Returns:
        (匹配成功的列表, 未匹配的列表)
    """
    if not symptoms:
        return [], []

    try:
        async with neo4j_driver.session() as session:
            result = await session.run(
                "MATCH (s:Symptom) WHERE s.name IN $names RETURN s.name AS name",
                names=symptoms,
            )
            records = await result.data()
            matched_set = {r["name"] for r in records}

        matched = [s for s in symptoms if s in matched_set]
        unmatched = [s for s in symptoms if s not in matched_set]
        logger.debug(f"Neo4j 精确匹配: 命中={matched}, 未命中={unmatched}")
        return matched, unmatched
    except Exception as e:
        logger.warning(f"Neo4j 症状匹配失败: {e}")
        return [], symptoms


async def match_milvus_layer3(unmatched: list[str], embedding_model, milvus_alias) -> tuple[dict[str, str], list[str]]:
    """
    第三层：Milvus 语义相似度兜底

    Args:
        unmatched: 未在 Neo4j 中命中的症状名
        embedding_model: 嵌入模型
        milvus_alias: Milvus 连接别名

    Returns:
        ({用户原词: 图谱标准词}, 仍然未匹配的列表)
    """
    if not unmatched:
        return {}, []

    from pymilvus import Collection

    mapped: dict[str, str] = {}
    still_unmatched: list[str] = []

    try:
        collection = Collection("symptoms", using=milvus_alias)
        collection.load()

        for symptom in unmatched:
            try:
                query_vec = embedding_model.embed_query(symptom)
                results = collection.search(
                    data=[query_vec],
                    anns_field="embedding",
                    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                    limit=1,
                    output_fields=["name"],
                )
                if results and results[0]:
                    top_hit = results[0][0]
                    score = top_hit.distance  # COSINE 相似度
                    if score >= SIMILARITY_THRESHOLD:
                        std_name = top_hit.entity.get("name", "")
                        mapped[symptom] = std_name
                        logger.debug(f"Milvus 语义映射: '{symptom}' → '{std_name}' (score={score:.3f})")
                    else:
                        still_unmatched.append(symptom)
                else:
                    still_unmatched.append(symptom)
            except Exception as e:
                logger.warning(f"Milvus 语义匹配异常 '{symptom}': {e}")
                still_unmatched.append(symptom)
    except Exception as e:
        logger.warning(f"Milvus 层整体失败: {e}")
        return {}, unmatched

    return mapped, still_unmatched


async def normalize_symptoms(user_input: str) -> dict:
    """
    完整三层症状标准化流水线入口

    用法:
        result = await normalize_symptoms("我拉肚子，浑身没劲")
        # result["all_standard"] → ["腹泻", "乏力"]
        # result["matched"] → ["腹泻", "乏力"]
        # result["mapped"] → {}
        # result["unmatched"] → []

    Returns:
        {
            "matched": ["腹泻"],         # 在第②层 Neo4j 直接命中的
            "mapped": {"拉肚子": "腹泻"}, # 在第③层 Milvus 语义映射的
            "unmatched": [],             # 三层都未匹配的
            "all_standard": ["腹泻"],    # matched + mapped 的值（可直接用于后续查询）
        }
    """
    from medical_agent.providers.llm import get_llm_qa
    from medical_agent.providers.embedding import get_embedding_model
    from medical_agent.infra.neo4j import get_neo4j_driver
    from medical_agent.infra.milvus import get_milvus_alias

    # ① LLM 提取
    llm = get_llm_qa()
    symptoms = await extract_symptoms_layer1(user_input, llm)
    if not symptoms:
        # LLM 失败时保留原始输入避免症状归零死循环
        fallback = user_input
        if isinstance(user_input, dict):
            fallback = user_input.get("content", str(user_input))
        logger.warning(f"[症状归一化] LLM提取失败，使用原始输入兜底: '{str(fallback)[:40]}'")
        return {"matched": [], "mapped": {}, "unmatched": [fallback],
                "all_standard": [fallback], "from_fallback": True}

    # ② Neo4j 精确匹配
    neo4j_driver = get_neo4j_driver()
    matched, unmatched = await match_neo4j_layer2(symptoms, neo4j_driver)

    # ③ Milvus 语义兜底
    embedding = get_embedding_model()
    alias = get_milvus_alias()
    if alias:
        mapped, still_unmatched = await match_milvus_layer3(unmatched, embedding, alias)
    else:
        mapped, still_unmatched = {}, unmatched

    all_standard = matched + list(mapped.values()) + still_unmatched

    return {
        "matched": matched,
        "mapped": mapped,
        "unmatched": still_unmatched,
        "all_standard": all_standard,
    }
