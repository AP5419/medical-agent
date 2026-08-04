# -*- coding: utf-8 -*-
"""分诊导诊智能体 - 症状采集、疾病推断、科室推荐、就诊指导"""

from typing import TypedDict, Annotated, Optional
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from loguru import logger

from medical_agent.providers.llm import get_llm_qa


# 系统提示词
INQUIRY_SYSTEM_PROMPT = """你是一位经验丰富的分诊导诊助手。你的职责是通过多轮对话收集用户症状，推断可能的疾病，并推荐合适的就诊科室。

## 工作流程
1. **加载患者信息**：查看患者历史就诊记录和过往症状
2. **紧急情况检测**：判断是否属于需要立即就医的紧急情况（胸痛、呼吸困难、意识不清、大出血等）
3. **症状提取**：从用户描述中提取关键症状信息（部位、性质、持续时间、诱因、伴随症状）
4. **候选疾病查询**：根据症状查询可能的疾病范围
5. **补充问询**：针对未明确的症状维度进行追问（疼痛程度、发作时间、加重缓解因素等）
6. **综合判断**：给出可能的疾病排名、置信度评估、就诊科室推荐

## 注意事项
- 每次只问1-2个关键问题，避免一次性追问过多
- 根据已收集的症状信息动态调整问题
- 对于紧急情况，优先建议拨打120或立即前往急诊
- 追问应聚焦于：部位、性质（胀痛/刺痛/绞痛）、持续时间、诱因、伴随症状、既往史
- 最终结论应包含：可能的疾病排名（按可能性排序）、推荐科室、紧急程度评估

## 免责声明
本导诊建议仅供参考，不能替代专业医生的诊断。如有紧急情况请立即拨打120。
"""


class InquiryState(TypedDict):
    """分诊导诊状态"""
    messages: Annotated[list, operator.add]
    symptoms: list
    normalized_symptoms: list
    symptom_detail: Optional[dict]
    candidate_diseases: list
    phase: str
    patient_context: Optional[dict]
    conclusion: Optional[str]
    handoff_payload: Optional[dict]
    symptom_rounds: int  # 症状采集轮次
    negation_count: int  # 连续否定回答次数
    symptom_detail_count: int  # 已收集的症状细节数
    in_detail_mode: bool  # 是否已进入细节收集状态（锁后永不回退）


def _load_patient(state: InquiryState) -> InquiryState:
    """节点1：加载患者信息和历史记录"""
    state["phase"] = "patient_loaded"
    return state


def _check_emergency(state: InquiryState) -> InquiryState:
    """节点2：紧急情况检测"""
    from medical_agent.orchestration.intent_router import IntentRouter

    router = IntentRouter()
    last_message = ""
    if state.get("messages"):
        last_msg = state["messages"][-1]
        if isinstance(last_msg, BaseMessage):
            last_message = last_msg.content
        else:
            last_message = str(last_msg)

    is_emergency, keywords = router.detect_emergency(last_message)
    if is_emergency:
        state["symptoms"] = state.get("symptoms", []) + [{"type": "emergency", "keywords": keywords}]
        state["phase"] = "emergency_detected"
    else:
        state["phase"] = "checked"
    return state


async def _extract_symptoms(state: InquiryState) -> InquiryState:
    """节点3：三层症状标准化管道

    ① LLM 口语→术语（如"拉肚子"→"腹泻"）
    ② Neo4j 精确匹配（验证术语在知识图谱中存在）
    ③ Milvus 语义兜底（向量相似度匹配未命中的词汇）
    """
    from medical_agent.agents.inquiry.symptom_normalizer import normalize_symptoms

    last_message = ""
    if state.get("messages"):
        last_msg = state["messages"][-1]
        if isinstance(last_msg, BaseMessage):
            last_message = last_msg.content
        elif isinstance(last_msg, dict):
            last_message = last_msg.get("content", "")
        else:
            last_message = str(last_msg)

    # 否定回答过滤：避免 "没有"/"A"/"B" 等被当症状，同时追踪连续否定次数
    _NEGATION_SET = {"没有","无","没","不","a","b","c","A","B","C",
                     "是","否","yes","no","不知道","不清楚","还行","还好","none"}
    if last_message and last_message.strip() in _NEGATION_SET:
        state["negation_count"] = state.get("negation_count", 0) + 1
        state["symptom_detail_count"] = state.get("symptom_detail_count", 0)
        state["normalized_symptoms"] = state.get("normalized_symptoms", [])
        state["symptom_detail"] = state.get("symptom_detail") or {}
        state["phase"] = "symptoms_extracted"
        return state

    # 非否定回答且已在细节模式时计为 detail（门控：未锁定时不计数）
    if last_message and len(last_message.strip()) > 1 and state.get("in_detail_mode"):
        state["negation_count"] = 0
        state["symptom_detail_count"] = state.get("symptom_detail_count", 0) + 1

    if last_message:
        # 运行三层标准化管道
        result = await normalize_symptoms(last_message)

        # 保存标准化结果供后续节点使用（累积去重）
        prev_normalized = state.get("normalized_symptoms", [])
        # from_fallback 标记: LLM 失败时兜底数据不进症状列表
        if result.get("from_fallback"):
            new_symptoms = []
        else:
            new_symptoms = result["all_standard"]
        state["normalized_symptoms"] = list(set(prev_normalized + new_symptoms))
        state["symptom_detail"] = {
            "matched": result["matched"],
            "mapped": result["mapped"],
            "unmatched": result["unmatched"],
        }

        # 保留旧格式兼容（供 _query_candidates 和 _conclude 使用）
        state["symptoms"] = state.get("symptoms", []) + result["all_standard"]
    else:
        state["normalized_symptoms"] = []
        state["symptom_detail"] = {"matched": [], "mapped": {}, "unmatched": []}

    state["phase"] = "symptoms_extracted"
    return state


def _query_candidates(state: InquiryState) -> InquiryState:
    """节点4：查询候选疾病 — 使用标准化症状"""
    raw_symptoms = state.get("symptoms", [])
    normalized = state.get("normalized_symptoms", [])

    # 优先使用标准化症状名
    symptom_names = normalized if normalized else [raw_symptoms] if isinstance(raw_symptoms, str) else raw_symptoms
    if not symptom_names:
        state["candidate_diseases"] = []
        state["phase"] = "candidates_queried"
        return state

    # 统一转字符串
    if isinstance(symptom_names, list):
        symptoms_text = "、".join([s if isinstance(s, str) else str(s) for s in symptom_names])
    else:
        symptoms_text = str(symptom_names)

    llm = get_llm_qa()

    query_prompt = f"""根据以下症状信息，列出最可能的5个候选疾病（按可能性从高到低排序）。

症状信息：{symptoms_text}

返回JSON格式：
{{"diseases": [{{"name": "疾病名", "probability": 0.0-1.0, "recommended_department": "科室名", "reasoning": "简要理由"}}]}}

仅返回JSON，不要其他内容。"""

    try:
        response = llm.invoke(query_prompt)
        import json
        import re
        text = response.content.strip()
        logger.info(f"[问诊] LLM候选疾病原始输出({len(text)}字): {text[:300]}")
        # 贪婪匹配最外层JSON（兼容嵌套花括号）
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            data = json.loads(json_match.group())
            state["candidate_diseases"] = data.get("diseases", [])
            logger.info(f"[问诊] 候选疾病={len(state['candidate_diseases'])}个: {[(d.get('name','?'), d.get('probability',0)) for d in state['candidate_diseases'][:3]]}")
        else:
            logger.warning(f"[问诊] _query_candidates: JSON解析失败，文本={text[:200]}")
    except Exception as e:
        logger.warning(f"[问诊] _query_candidates失败: {e}")

    state["phase"] = "candidates_queried"
    return state


def _ask_questions(state: InquiryState) -> InquiryState:
    """节点5：补充问询——针对未明确的维度追问"""
    symptoms = state.get("symptoms", [])
    candidates = state.get("candidate_diseases", [])

    if state.get("phase") == "emergency_detected":
        state["conclusion"] = (
            "⚠️ 紧急提醒：您描述的症状包含紧急情况关键词，"
            "请立即拨打120急救电话或前往最近的医院急诊科就诊。"
            "在等待救援期间，请保持镇静，如有可能请让他人陪护。"
        )
        state["phase"] = "concluded"
        return state

    # 检查是否已收集足够信息（至少2轮症状采集）
    if len(symptoms) >= 2 or len(state.get("messages", [])) >= 4:
        state["phase"] = "ready_to_conclude"
        return state

    llm = get_llm_qa()
    symptoms_text = str(symptoms[-1]) if symptoms else ""

    ask_prompt = f"""作为分诊导诊助手，用户已描述的当前症状：{symptoms_text}

请提出1-2个最关键的补充问询问题，帮助进一步明确症状。问题应聚焦于：
- 症状的具体部位和性质
- 持续时间和发作频率
- 加重或缓解因素
- 伴随症状
- 既往病史

只返回问询问题，不要额外解释。"""

    try:
        response = llm.invoke(ask_prompt)
        follow_up_question = response.content.strip()
        state["handoff_payload"] = {"type": "follow_up", "question": follow_up_question}
    except Exception:
        state["handoff_payload"] = {"type": "follow_up", "question": "请问您的症状持续多久了？有没有其他伴随症状？"}

    state["phase"] = "asking"
    return state


def _prob_label(p: float) -> str:
    """概率→中文可读标签"""
    if p >= 0.8:
        return "可能性极高"
    if p >= 0.6:
        return "可能性大"
    if p >= 0.4:
        return "中等可能"
    if p >= 0.2:
        return "略有可能"
    return "可能性较低"


def _conclude(state: InquiryState) -> InquiryState:
    """节点6：结构化分诊结论——优先用候选疾病数据拼装，LLM仅作异常兜底"""
    normalized = state.get("normalized_symptoms", [])
    logger.info(f"[问诊] 收尾: 症状={normalized}")
    candidates = state.get("candidate_diseases", [])
    symptom_detail = state.get("symptom_detail", {})

    if not candidates and not normalized:
        state["conclusion"] = "根据您提供的信息，暂时无法做出明确判断。建议您补充更多症状细节，或前往综合医院全科/内科就诊。"
        state["phase"] = "concluded"
        return state

    symptoms_text = "、".join(normalized) if normalized else "未提供"

    # 结构化候选疾病文本（保留标签、科室、理由）
    candidate_lines = []
    for d in candidates[:5]:
        name = d.get("name", "?")
        prob = d.get("probability", 0)
        dept = d.get("recommended_department", "")
        reason = d.get("reasoning", "")
        candidate_lines.append(f"{name}（{_prob_label(prob)}，{dept}）：{reason}")
    candidates_text = "\n".join(candidate_lines) if candidate_lines else "无"

    unmatched_note = ""
    unmatched = symptom_detail.get("unmatched", [])
    if unmatched:
        unmatched_note = f"\n注意：以下描述未匹配到标准术语：{'、'.join(unmatched)}，仅供参考。"

    # 推断推荐科室（取top1候选的科室）
    top_dept = candidates[0].get("recommended_department", "内科") if candidates else "内科"

    # 按科室动态生成紧急提醒
    _DEPT_WARNINGS = {
        "骨科": "如关节红肿剧痛、无法承重或行走困难，请立即急诊",
        "消化内科": "如出现黑便、呕血或剧烈难忍的腹痛，请立即急诊",
        "心血管内科": "如出现持续胸痛、呼吸困难或意识不清，请立即拨打120",
        "呼吸内科": "如出现呼吸困难、唇色发紫或持续高烧不退，请立即急诊",
        "神经内科": "如出现剧烈头痛、肢体瘫痪或意识模糊，请立即急诊",
        "神经外科": "如出现剧烈头痛、呕吐或意识障碍，请立即急诊",
        "泌尿外科": "如出现肉眼血尿、剧烈腰痛或排尿困难，请立即急诊",
        "感染科": "如出现持续高烧不退、意识模糊或呼吸困难，请立即急诊",
        "急诊科": "如症状危及生命，请立即拨打120",
    }
    warning = _DEPT_WARNINGS.get(top_dept, "如症状突然加重，请立即急诊")

    conclude_prompt = f"""你是分诊导诊助手。请仅根据以下候选疾病数据生成结论，不要添加数据之外的推断，不要寒暄、不要称呼。

症状：{symptoms_text}{unmatched_note}
候选疾病：
{candidates_text}

严格按以下格式输出（不超过300字，不要用markdown标题）:

分析完成。症状：{symptoms_text}

可能的疾病诊断：
1. {{疾病名}} {{匹配程度}}
2. {{疾病名}} {{匹配程度}}
...

推荐科室：{top_dept}
紧急提醒：{warning}
免责声明：AI分诊仅供就医参考，不能替代医生诊断。请前往正规医院就诊。"""

    llm = get_llm_qa()
    try:
        response = llm.invoke(conclude_prompt)
        state["conclusion"] = response.content.strip()
    except Exception:
        # LLM 失败时直接用结构化数据拼装
        candidate_summary = "\n".join(
            f"{i+1}. {d.get('name','?')}  {_prob_label(d.get('probability',0))}"
            for i, d in enumerate(candidates[:5])
        )
        state["conclusion"] = (
            f"分析完成。症状：{symptoms_text}\n\n"
            f"可能的疾病诊断：\n{candidate_summary}\n\n"
            f"推荐科室：{top_dept}\n"
            f"紧急提醒：{warning}\n"
            f"免责声明：AI分诊仅供就医参考，不能替代医生诊断。请前往正规医院就诊。"
        )

    state["messages"] = state.get("messages", []) + [{"role": "assistant", "content": state["conclusion"]}]
    state["phase"] = "concluded"
    return state


def _save_record(state: InquiryState) -> InquiryState:
    """节点7：保存就诊记录"""
    state["phase"] = "record_saved"
    return state


def _route_after_symptoms(state: InquiryState) -> str:
    """症状提取后路由：走企业级收敛检查流程"""
    return "quick_check"


def _should_conclude(state: InquiryState) -> tuple[bool, str]:
    """企业级三层收敛判断：加权综合评分"""
    rounds = state.get("symptom_rounds", 0)
    normalized = state.get("normalized_symptoms", [])
    candidates = state.get("candidate_diseases", [])

    if rounds >= 5:
        return True, f"轮次={rounds}>=5 强制停止"
    top1_prob = candidates[0].get("probability", 0) if candidates and isinstance(candidates[0], dict) else 0
    top2_prob = candidates[1].get("probability", 0) if len(candidates) > 1 and isinstance(candidates[1], dict) else 0
    confidence_ok = top1_prob >= 0.5 and (top1_prob - top2_prob) >= 0.15
    symptoms_ok = len(normalized) >= 3
    rounds_ok = rounds >= 2
    score = symptoms_ok * 0.4 + confidence_ok * 0.3 + rounds_ok * 0.3
    logger.info(f"[问诊] 收敛评分={score:.1f} (维度={symptoms_ok}({len(normalized)}) 置信度={confidence_ok}({top1_prob:.2f}/{top2_prob:.2f}) 轮次={rounds_ok}({rounds}))")
    if score >= 0.6:
        return True, f"评分{score:.1f}>=0.6"
    return False, f"评分{score:.1f}<0.6"


def _quick_check(state: InquiryState) -> InquiryState:
    """企业级收敛检查：先 detail 收敛，再 3 因子评分"""
    state["symptom_rounds"] = state.get("symptom_rounds", 0) + 1
    normalized = state.get("normalized_symptoms", [])
    if len(normalized) >= 2:
        state = _query_candidates(state)

    negation_count = state.get("negation_count", 0)
    detail_count = state.get("symptom_detail_count", 0)
    rounds = state.get("symptom_rounds", 1)

    # 否定≥2 → 锁定细节模式（粘性，永不回退）
    if negation_count >= 2:
        state["in_detail_mode"] = True

    # 细节收敛优先于 3 因子评分（避免 rounds>=5 强制停止绕过补查）
    if detail_count >= 2 and rounds >= 3:
        logger.info(f"[问诊] 细节收敛: detail={detail_count}, rounds={rounds}")
        if not state.get("candidate_diseases"):
            state = _query_candidates(state)
        state["phase"] = "ready_to_conclude"
        state["handoff_payload"] = None
        return state

    should_stop, reason = _should_conclude(state)
    logger.info(f"[问诊] 收敛: {reason}")
    if should_stop:
        state["phase"] = "ready_to_conclude"
        state["handoff_payload"] = None
        return state

    candidates = state.get("candidate_diseases", [])
    llm = get_llm_qa()
    symptoms_text = "、".join(normalized) if normalized else "未明确"

    # 自适应追问策略
    if state.get("in_detail_mode"):
        # 分支②：细节模式（持久）→ 追问症状细节，注入上一轮问答复避免重复
        last_user = ""
        last_assistant = ""
        for m in reversed(state.get("messages", [])):
            c = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
            r = m.get("role", "") if isinstance(m, dict) else getattr(m, "type", "")
            if not last_user and r in ("user", "human"):
                last_user = c
            if not last_assistant and r in ("assistant", "ai"):
                last_assistant = c
            if last_user and last_assistant:
                break
        context = ""
        if last_assistant and last_user:
            context = f"上次已问: {last_assistant[:50]}，答: {last_user[:50]}。"
        prompt = (
            f"已知症状：{symptoms_text}。患者已确认无其他伴随症状。"
            f"{context}"
            f"请追问一个尚未问过的细节（如持续时间/疼痛程度/加重缓解因素），"
            f"给出2-3个选项。只输出问题本身。"
        )
    else:
        # 分支①：伴随模式（默认）
        cand_hint = ""
        if len(candidates) >= 2:
            cand_hint = (
                f"候选: {candidates[0].get('name','')}({candidates[0].get('probability',0):.0%}) vs "
                f"{candidates[1].get('name','')}({candidates[1].get('probability',0):.0%})。"
            )
        prompt = (
            f"已知症状：{symptoms_text}。" +
            (f"{cand_hint}" if cand_hint else "") +
            "请用一句话追问1个最关键的问题。优先问患者有**没有**其他伴随症状" +
            "（如恶心/发热/腹泻/乏力/头晕等），而非追问已有症状的细节。" +
            "给出2-3个选项降低用户输入难度。只输出问题本身。"
        )

    try:
        resp = llm.invoke(prompt)
        question = resp.content.strip()
    except Exception:
        question = "请问症状持续多久了？有没有其他伴随情况？"
    logger.info(f"[问诊] 追问: {question}")
    state["handoff_payload"] = {"type": "follow_up", "question": question}
    state["messages"] = state.get("messages", []) + [{"role": "assistant", "content": question}]
    state["phase"] = "asking"
    return state


def _route_after_check(state: InquiryState) -> str:
    """紧急检测后的路由判断"""
    if state.get("phase") == "emergency_detected":
        return "conclude"
    return "extract"


def _route_after_question(state: InquiryState) -> str:
    """问询后的路由判断"""
    if state.get("phase") == "ready_to_conclude":
        return "conclude"
    return END


# ── 带检查点的 Agent 单例（多轮有状态对话） ─────────────────────────────────

_inquiry_agent_cache = None


def _get_inquiry_agent():
    """获取或创建带 MemorySaver 检查点的问诊 Agent（单例）

    MemorySaver 在进程内存中保存对话状态，生产环境替换为 RedisSaver。
    """
    global _inquiry_agent_cache
    if _inquiry_agent_cache is None:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        _inquiry_agent_cache = create_inquiry_agent(checkpointer=checkpointer)
        logger.info("[问诊] Agent 已创建（MemorySaver 检查点）")
    return _inquiry_agent_cache


def create_inquiry_agent(checkpointer=None):
    """创建分诊导诊智能体——优化版StateGraph工作流

    全部: load_patient → check_emergency → extract_symptoms(3层) → quick_check(企业级收敛+候选疾病查询) → conclude(生成结论) → END

    Args:
        checkpointer: LangGraph 检查点保存器（MemorySaver / RedisSaver），用于多轮状态持久化
    """
    workflow = StateGraph(InquiryState)

    # 添加节点（query_candidates 已内嵌在 _quick_check 中，不再作为独立节点）
    workflow.add_node("load_patient", _load_patient)
    workflow.add_node("check_emergency", _check_emergency)
    workflow.add_node("extract_symptoms", _extract_symptoms)
    workflow.add_node("quick_check", _quick_check)
    workflow.add_node("conclude", _conclude)
    workflow.add_node("save_record", _save_record)

    # 设置入口
    workflow.set_entry_point("load_patient")

    # 添加边
    workflow.add_edge("load_patient", "check_emergency")

    # 紧急检测后的条件路由
    workflow.add_conditional_edges(
        "check_emergency",
        _route_after_check,
        {"conclude": "conclude", "extract": "extract_symptoms"},
    )

    # 症状提取后统一走简化流程
    workflow.add_edge("extract_symptoms", "quick_check")

    # 简化检查后的路由
    workflow.add_conditional_edges(
        "quick_check",
        _route_after_question,
        {"conclude": "conclude", END: END},
    )

    workflow.add_edge("conclude", "save_record")
    workflow.add_edge("save_record", END)

    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


async def run_inquiry(message: str, session_id: str = "", patient_context: Optional[dict] = None) -> str:
    """运行分诊导诊流程（多轮有状态）

    同一 session_id 的多次调用共享对话状态：
    - 首轮：创建初始状态，开始症状采集
    - 后续轮次：基于检查点恢复，累积症状直到收敛

    收敛后下一轮自动重置为新问诊。

    Args:
        message: 用户症状描述
        session_id: 会话ID，用于跨轮状态持久化
        patient_context: 患者上下文信息（可选）

    Returns:
        分诊导诊结论文本或追问问题
    """
    agent = _get_inquiry_agent()
    thread_id = f"inquiry:{session_id}" if session_id else f"inquiry:default"
    config = {"configurable": {"thread_id": thread_id}}
    logger.info(f"[问诊] 开始: '{message[:40]}' (session={session_id[:16] if session_id else 'none'})")

    # 检查是否已有检查点（后续轮次）
    try:
        existing = await agent.aget_state(config)
        is_first_turn = existing is None or not existing.values
    except Exception:
        is_first_turn = True

    if is_first_turn:
        # 首轮：完整初始状态
        initial_state = {
            "messages": [{"role": "user", "content": message}],
            "symptoms": [],
            "normalized_symptoms": [],
            "symptom_detail": None,
            "candidate_diseases": [],
            "phase": "initial",
            "patient_context": patient_context or {},
            "conclusion": None,
            "handoff_payload": None,
            "symptom_rounds": 0,
            "negation_count": 0,
            "symptom_detail_count": 0,
            "in_detail_mode": False,
        }
    else:
        # 后续轮次：仅追加消息，其余状态从检查点恢复
        initial_state = {"messages": [{"role": "user", "content": message}]}

    result = await agent.ainvoke(initial_state, config=config)

    # 优先返回追问问题
    handoff = result.get("handoff_payload", {})
    if handoff and handoff.get("type") == "follow_up":
        return handoff.get("question", "")
    # 返回结论
    conclusion = result.get("conclusion", "")
    if conclusion:
        return conclusion
    # 返回最后一条 AI 消息
    messages = result.get("messages", [])
    for m in reversed(messages):
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        role = m.get("role", "") if isinstance(m, dict) else getattr(m, "type", "")
        if content and role not in ("user", "human"):
            logger.info(f"[问诊] 结束: 返回 {len(content)} 字符")
            return content
    logger.info("[问诊] 结束: 无有效回复")
    return "分诊导诊流程已完成，请查看结果。"
