# -*- coding: utf-8 -*-
"""主管编排 - 第3层：意图路由 + Worker调度（无ReAct，确定性编排）"""

from typing import Optional

from loguru import logger

from medical_agent.core.config import get_settings
from medical_agent.orchestration.intent_router import IntentRouter, IntentType


# ── Worker Agent 调用映射 ────────────────────────────────────────────────

_AGENT_CACHE = {}  # {intent: agent}

_EMERGENCY_PREFIX = (
    "⚠️ 紧急提醒：根据您的描述，建议立即拨打120急救电话或前往最近的医院急诊科。"
    "在等待救援期间，请保持镇静。\n\n"
)


async def _get_or_create_agent(intent: str):
    """懒加载获取 Agent 实例（单例缓存）"""
    if intent in _AGENT_CACHE:
        return _AGENT_CACHE[intent]

    if intent == "inquiry":
        from medical_agent.agents.inquiry.agent import run_inquiry
        _AGENT_CACHE[intent] = run_inquiry
    elif intent == "report":
        from medical_agent.agents.report.agent import get_report_agent
        _AGENT_CACHE[intent] = await get_report_agent()
    elif intent == "drug":
        from medical_agent.agents.drug.agent import get_drug_agent
        _AGENT_CACHE[intent] = await get_drug_agent()
    elif intent == "knowledge":
        from medical_agent.agents.knowledge.agent import get_knowledge_agent
        _AGENT_CACHE[intent] = await get_knowledge_agent()
    elif intent == "operation":
        from medical_agent.agents.operation.agent import get_operation_agent
        _AGENT_CACHE[intent] = await get_operation_agent()
    else:
        _AGENT_CACHE[intent] = None

    return _AGENT_CACHE[intent]


async def _call_worker(intent: str, message: str, session_id: str = "", role: str = "") -> str:
    """调用 Worker Agent（确定性，无 ReAct）"""
    agent_or_fn = await _get_or_create_agent(intent)
    if agent_or_fn is None:
        return "未能理解您的问题，请换一种方式描述。"
    if callable(agent_or_fn):  # inquiry 直接是 async function
        return await agent_or_fn(message, session_id=session_id)
    if intent == "report":
        return await _handle_report(message)
    if intent == "drug":
        return await _handle_drug(message, role=role)
    result = await agent_or_fn.ainvoke({"messages": [{"role": "user", "content": message}]})
    msgs = result.get("messages", [])
    for m in reversed(msgs):
        content = getattr(m, "content", "") or ""
        if content and getattr(m, "type", "") != "human":
            return content
    return "抱歉，暂时无法处理您的问题。"


# ── 报告解读确定性 Pipeline（替代 ReAct）───────────────────────────────────

# 报告类型关键词映射（用户口语→LISAdapter mock 数据枚举值）
_REPORT_TYPE_MAP = {
    "血常规": "血常规", "血象": "血常规", "血": "血常规",
    "生化": "生化检查", "肝功能": "生化检查", "肾功能": "生化检查", "血糖": "生化检查",
    "尿常规": "尿常规", "尿": "尿常规",
}

# Mock 数据中已知的患者名
_KNOWN_PATIENTS = ["张三", "李四"]


async def _parse_report_query(message: str) -> tuple[str, str]:
    """从用户消息中提取患者名和报告类型（规则优先，LLM 兜底）"""
    # ① 关键词提取患者名
    patient_name = ""
    for p in _KNOWN_PATIENTS:
        if p in message:
            patient_name = p
            break

    # ② 关键词提取报告类型
    report_type = "all"
    for kw, mapped in _REPORT_TYPE_MAP.items():
        if kw in message:
            report_type = mapped
            break

    # ③ 关键词未命中 → LLM 1 次提取
    if not patient_name:
        try:
            from medical_agent.providers.llm import get_llm_qa
            llm = get_llm_qa()
            prompt = f"从以下消息中提取患者姓名（仅返回姓名，无则返回空）：{message[:200]}"
            resp = llm.invoke(prompt)
            patient_name = resp.content.strip()
        except Exception:
            pass

    return patient_name, report_type


async def _fetch_report_data(patient_name: str, report_type: str) -> list:
    """直接调 LISAdapter 获取报告，不经过 LLM 决策"""
    if not patient_name:
        return []
    from medical_agent.adapters.lis import LISAdapter
    return await LISAdapter.search_reports(patient_name=patient_name, report_type=report_type)


async def _format_report_output(patient: str, report_type: str, reports: list) -> str:
    """报告解读输出：多份报告→程序算趋势+LLM总结；单份→LLM格式化"""
    if not reports:
        return f"未找到患者 {patient} 的检验报告。请确认姓名是否正确（当前仅支持：张三-血常规/生化、李四-尿常规）。"

    from medical_agent.providers.llm import get_llm_qa

    # ── 多份报告：程序计算趋势 → LLM 总结（1 次调用）──
    if len(reports) > 1:
        # ① 构建时序数据
        indicator_series = {}
        for r in sorted(reports, key=lambda x: x.get("report_date", "")):
            for ind in r.get("indicators", []):
                name = ind["name"]
                if name not in indicator_series:
                    indicator_series[name] = []
                indicator_series[name].append(ind)

        # ② 程序计算趋势（确定性，不依赖 LLM）
        trend_lines = []
        for name, points in indicator_series.items():
            if len(points) < 2:
                trend_lines.append(f"  {name}: {points[0]['value']}{points[0]['unit']} ({points[0]['status']})")
                continue

            # 值序列
            values = [p["value"] for p in points]
            dates = [p.get("date","") or r.get("report_date","")[-5:] for p, r in
                     zip(points, sorted(reports, key=lambda x: x.get("report_date","")))]

            # 方向判断（first → last）
            first, last = values[0], values[-1]
            if not isinstance(first, (int, float)) or not isinstance(last, (int, float)):
                path = " → ".join(str(v) for v in values)
                trend_lines.append(f"  {name}: {path} (-)")
                continue

            direction = "↓" if last < first else "↑" if last > first else "→"

            # 末次值是否在参考范围内（程序解析 ref_range）
            ref = points[-1].get("ref_range", "")
            last_in_range = False
            if ref and isinstance(last, (int, float)):
                try:
                    ref = ref.replace("<", "").strip()
                    if "-" in ref:
                        lo, hi = ref.split("-")
                        last_in_range = float(lo) <= last <= float(hi)
                    else:
                        last_in_range = last <= float(ref)
                except (ValueError, AttributeError):
                    pass

            path = " → ".join(str(v) for v in values)
            status = "已正常" if last_in_range else "仍略高" if direction == "↓" else "仍异常"
            trend_lines.append(f"  {name}: {path}  {status}")

        trends_text = "\n".join(trend_lines)

        # ③ LLM 1 次总结（输入只有趋势表 + 入院背景，不是完整原始数据）
        admission = await _get_admission_context(patient)
        context = ""
        if admission:
            context = (f"患者 {admission.get('patient_name',patient)}，{admission.get('gender','')}，"
                       f"{admission.get('age','')}岁。{admission.get('admission_diagnosis','')}。"
                       f"住院 {admission.get('admission_date','')}-{admission.get('discharge_date','')}，"
                       f"予 {admission.get('treatment','')}。")

        prompt = f"""仅根据以下数据生成总结，不要添加推断，不要寒暄。

{context}

实验室趋势：
{trends_text}

请用3-4句话总结治疗反应和当前状态（须包含首次和末次具体数值），仍略高或仍异常的指标请用 **粗体** 标注，不超过200字。
格式: 分析完成。{patient}住院期间检验趋势：[...总结...]
建议：[...]
免责声明：AI分析仅供参考，请以临床医生诊断为准。"""

        llm = get_llm_qa()
        try:
            resp = llm.invoke(prompt)
            return resp.content.strip()
        except Exception:
            return (f"分析完成。{patient}住院期间检验趋势：\n{trends_text}\n\n"
                    f"免责声明：AI分析仅供参考，请以临床医生诊断为准。")

    # ── 单份报告：LLM 格式化 ──
    report_items = []
    abnormal_items = []
    for r in reports:
        indicators_text = []
        for i in r.get("indicators", []):
            line = f"  {i['name']}: {i['value']}{i['unit']} (参考: {i['ref_range']}) [{i['status']}]"
            indicators_text.append(line)
            if i.get("status") not in ("正常",):
                abnormal_items.append(line)
        report_items.append(f"报告ID: {r['id']}\n类型: {r['report_type']}\n日期: {r['report_date']}\n" +
                             "\n".join(indicators_text))

    reports_text = "\n\n".join(report_items)
    abnormal_text = "\n".join(abnormal_items) if abnormal_items else "无"

    prompt = f"""仅根据以下检验报告数据生成解读，不要添加数据之外的推断，不要寒暄、不要称呼。

患者：{patient}
{reports_text}

异常指标：
{abnormal_text}

严格按以下格式输出（不超过300字）：
分析完成。{patient}，{report_type}

异常指标：
- {{指标名}} {{值}}{{单位}}（{{状态}}）：{{简要临床意义}}（异常指标名和值请用 **粗体** 标注）
...
综合意见：{{1-2句总结}}
免责声明：AI解读仅供参考，请以临床医生诊断为准。"""

    llm = get_llm_qa()
    try:
        resp = llm.invoke(prompt)
        return resp.content.strip()
    except Exception:
        return f"分析完成。{patient}，{report_type}\n\n异常指标：\n{abnormal_text}\n\n综合意见：建议咨询临床医生\n免责声明：AI解读仅供参考，请以临床医生诊断为准。"


async def _get_admission_context(patient_name: str) -> dict:
    """获取入院背景信息（供趋势分析用）"""
    from medical_agent.adapters.lis import LISAdapter
    return await LISAdapter.get_admission_info(patient_name=patient_name)


async def _handle_report(message: str) -> str:
    """报告解读确定性 pipeline：出院小结 vs 单次查询"""
    patient, report_type = await _parse_report_query(message)

    # 出院小结/摘要关键词 → 走聚合路径
    _DISCHARGE_KEYS = ["出院小结", "出院总结", "住院总结", "出院摘要", "住院摘要"]
    if any(k in message for k in _DISCHARGE_KEYS):
        return await _handle_discharge_summary(patient)

    # 默认：单次查询
    logger.info(f"[报告] 患者={patient}, 类型={report_type}")
    reports = await _fetch_report_data(patient, report_type)
    logger.info(f"[报告] 查到 {len(reports)} 条记录")
    return await _format_report_output(patient, report_type, reports)


async def _handle_discharge_summary(patient_name: str) -> str:
    """出院小结自动生成：程序提取结构化时间线 → LLM 生成诊疗经过（1次调用）→ 程序组装全文"""
    from medical_agent.adapters.lis import LISAdapter
    from medical_agent.providers.llm import get_llm_qa

    if not patient_name:
        return "请提供患者姓名以生成出院小结。"

    admission = await LISAdapter.get_admission_info(patient_name=patient_name)
    if not admission:
        return f"未找到患者 {patient_name} 的住院信息，无法生成出院小结。"

    # ① 获取全部 LIS 数据并按日期排序
    lis_data = await LISAdapter.get_patient_lis_report(patient_name=patient_name)
    reports = sorted(lis_data.get("reports", []), key=lambda r: r.get("report_date", ""))
    if not reports:
        return f"未找到患者 {patient_name} 的住院检验数据。"

    # ② 程序提取结构化治疗时间线（确定性，不依赖 LLM）
    milestones = []
    for r in reports:
        labs = {}
        for ind in r.get("indicators", []):
            labs[ind["name"]] = f"{ind['value']}{ind['unit']}"
        milestone = {
            "date": r["report_date"][-5:],
            "day": _compute_day(admission["admission_date"], r["report_date"]),
            "labs": labs,
        }
        milestones.append(milestone)

    # ③ 构建时间线文本（供 LLM 使用）
    tl_lines = []
    for m in milestones:
        tl_lines.append(f"{m['day']} ({m['date']}): {', '.join(f'{k} {v}' for k, v in m['labs'].items())}")
    timeline_text = "\n".join(tl_lines)

    course_prompt = f"""请根据以下结构化治疗时间线生成诊疗经过段落。按时间顺序组织，不编造数据中不存在的事件。

患者：{admission['patient_name']}（{admission['gender']}，{admission['age']}岁）
入院诊断：{admission['admission_diagnosis']}
入院日期：{admission['admission_date']}，出院日期：{admission['discharge_date']}
治疗方案：{admission['treatment']}
影像学检查：{admission['CT_summary']}

实验室检查时间线：
{timeline_text}

请生成1段诊疗经过（不超过250字），包含：
- 入院后启动的治疗方案
- 治疗过程中各项指标随时间的变化趋势
- 影像学检查结果
- 出院时状态

只输出诊疗经过正文，不加标题。"""

    llm = get_llm_qa()
    try:
        resp = llm.invoke(course_prompt)
        admission_course = resp.content.strip()
    except Exception:
        # LLM 失败 → 简单拼接
        course_parts = [f"入院后予{admission['treatment']}。"]
        for m in milestones:
            key_labs = ", ".join(f"{k} {v}" for k, v in list(m["labs"].items())[:4])
            course_parts.append(f"{m['date']}：{key_labs}。")
        admission_course = "".join(course_parts)

    # ④ 程序组装出院小结全文（模板填充，不调 LLM）
    hospital = admission.get("hospital_name", "")
    dc_lines = [
        hospital,
        "出院小结",
        "",
        f"姓名：{admission['patient_name']}  性别：{admission['gender']}  "
        f"年龄：{admission['age']}  住院号：{admission['admission_number']}",
        f"入院日期：{admission['admission_date']}  出院日期：{admission['discharge_date']}",
        "",
        f"入院情况：患者因\"{admission['chief_complaint']}\"入院。入院查体：{admission['vital_signs']}。",
        f"入院诊断：{admission['admission_diagnosis']}",
        f"诊疗经过：{admission_course}",
        f"出院诊断：{admission['discharge_diagnosis']}",
        f"出院情况：{admission.get('discharge_condition', '')}",
        f"出院医嘱：{admission['discharge_orders']}",
        "",
        "（AI辅助生成，待主管医师审核签字）",
    ]
    return "\n".join(dc_lines)


def _compute_day(admission_date: str, report_date: str) -> str:
    """计算住院第几天（入院日为 D0）"""
    try:
        from datetime import datetime
        adm = datetime.strptime(admission_date, "%Y-%m-%d")
        rep = datetime.strptime(report_date, "%Y-%m-%d")
        day = (rep - adm).days
        return f"入院" if day == 0 else f"D{day}" if day > 0 else f"D{day}"
    except Exception:
        return report_date


# ── 药物咨询确定性 Pipeline（药师处方审核 + 医生药品查询）─────────────────

async def _handle_drug(message: str, role: str = "") -> str:
    """药物咨询确定性 pipeline：药师→审核+查询，医生→仅查询"""
    patient_name = _extract_patient_name(message)

    # 处方审核关键词 → 仅药师可操作
    _REVIEW_KEYS = ["审核", "审查", "处方", "核对", "检查处方", "审方"]
    if any(k in message for k in _REVIEW_KEYS) and patient_name:
        if role != "pharmacist":
            return "处方审核仅限药师操作。医生请使用药品查询功能，患者请咨询医师。"
        return await _review_prescription(patient_name)

    # 默认：药品查询
    return await _query_drug_info(message, patient_name)


def _extract_patient_name(message: str) -> str:
    """从消息中提取患者姓名"""
    for p in _KNOWN_PATIENTS:
        if p in message:
            return p
    return ""


async def _review_prescription(patient_name: str) -> str:
    """药师处方审核：检查相互作用 + 过敏史 + 检验指标（1 次 LLM）"""
    from medical_agent.adapters.his import HISAdapter
    from medical_agent.providers.llm import get_llm_qa

    if not patient_name:
        return "请提供患者姓名以进行处方审核（如：审核张三的处方）。"

    # ① 获取患者信息
    patient_info = await HISAdapter.get_patient_info(patient_name)
    if not patient_info or not patient_info.get("prescriptions"):
        return f"未找到患者 {patient_name} 的处方记录。"

    prescriptions = patient_info["prescriptions"]
    pending = [p for p in prescriptions if p.get("status") in ("待审核", "待取药")]
    target = pending if pending else prescriptions  # 优先审核未完成的

    # ② 提取所有药品名
    all_drugs = set()
    for p in target:
        for d in p.get("drugs", []):
            all_drugs.add(d["name"])

    # ③ 逐对检查相互作用
    drug_list = list(all_drugs)
    interaction_text = "未检测到显著相互作用"
    if len(drug_list) >= 2:
        try:
            from medical_agent.engines.graph.graph_rag import GraphRAGEngine
            rag = GraphRAGEngine()
            interactions = []
            for i in range(len(drug_list)):
                for j in range(i + 1, len(drug_list)):
                    result = await rag.check_drug_interaction(drug_list[i], drug_list[j])
                    if result and "无" not in str(result):
                        evidence = "指南/明确" if any(k in str(result) for k in ("明确","显著","禁忌","CYP")) else "文献/潜在"
                        interactions.append(f"  {drug_list[i]} + {drug_list[j]}（{evidence}）: {result}")
            if interactions:
                interaction_text = "\n".join(interactions)
        except Exception:
            interaction_text = "（相互作用检查服务不可用）"

    # ④ 过敏史检查
    allergies = patient_info.get("allergies", [])
    allergy_warning = ""
    for d in all_drugs:
        for allergy in allergies:
            if allergy in d:
                allergy_warning += f"\n  ⚠ {d}：患者有{allergy}过敏史！"

    # ⑤ 查检验指标（药物-检验联动，取每个指标最近一次测量值）
    lab_context = ""
    try:
        from medical_agent.adapters.lis import LISAdapter
        lab_data = await LISAdapter.get_patient_lis_report(patient_name=patient_name)
        if lab_data and lab_data.get("reports"):
            TARGET = ("肌酐", "血糖", "超敏C反应蛋白")
            latest_by_name = {}
            for r in sorted(lab_data["reports"], key=lambda x: x.get("report_date", "")):
                for ind in r.get("indicators", []):
                    if ind["name"] in TARGET:
                        latest_by_name[ind["name"]] = ind
            for ind in latest_by_name.values():
                lab_context += f"\n  检验: {ind['name']} {ind['value']}{ind['unit']} (参考{ind['ref_range']}) [{ind['status']}]"
    except Exception:
        pass

    # ⑥ LLM 1 次生成审核报告
    prescription_lines = []
    for p in target:
        drugs_str = ", ".join(f"{d['name']} {d.get('dose','')}" for d in p.get("drugs", []))
        prescription_lines.append(f"  {p['prescription_id']} ({p['department']}): {drugs_str} [{p['status']}]")

    prompt = f"""你是药学审核助手。仅根据以下数据生成处方审核报告，不要添加数据之外的推断。

患者：{patient_name}
过敏史：{', '.join(allergies) if allergies else '无'}
当前处方：
{chr(10).join(prescription_lines)}

相互作用检查：
{interaction_text}
{allergy_warning}
{lab_context}

按以下格式输出（不超过300字）：
处方审核报告

患者：{patient_name}
过敏史：{', '.join(allergies) if allergies else '无'}
{lab_context}

审核处方：
{chr(10).join(prescription_lines)}

相互作用：{interaction_text}
{allergy_warning}

审核意见：{{1-2句话，是否通过审核}}
（AI辅助审核，请药师最终确认）"""

    llm = get_llm_qa()
    try:
        resp = llm.invoke(prompt)
        return resp.content.strip()
    except Exception:
        return "\n".join([
            f"处方审核报告\n\n患者：{patient_name}",
            f"过敏史：{', '.join(allergies) if allergies else '无'}",
            f"\n审核处方：\n{chr(10).join(prescription_lines)}",
            f"\n相互作用：\n{interaction_text}{allergy_warning}",
            "\n审核意见：请药师人工审核确认",
            "（AI辅助审核，请药师最终确认）",
        ])


async def _extract_drug_name(message: str) -> str:
    """LLM 从用户消息中提取药品名称（处理缩略名、商品名等变体）"""
    from medical_agent.adapters.his import HISAdapter
    from medical_agent.providers.llm import get_llm_qa

    drug_names = [d["name"] for d in HISAdapter.MOCK_DRUGS]
    drug_aliases = []
    for d in HISAdapter.MOCK_DRUGS:
        drug_aliases.extend(d.get("aliases", []))

    prompt = f"""从用户消息中提取药品名称。用户可能使用缩略名、商品名或通用名。

用户消息：{message}

已知药品（仅供参考，用户可能使用变体名称）：
{', '.join(drug_names + drug_aliases)}

如果用户提及了药品，返回：{{"drug": "提取到的药品名", "query_type": "info"}}
如果用户未提及具体药品，返回：{{"drug": "", "query_type": "unknown"}}
仅返回JSON，不要其他内容。"""

    llm = get_llm_qa()
    try:
        resp = llm.invoke(prompt)
        import json, re
        text = resp.content.strip()
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            data = json.loads(match.group())
            return data.get("drug", "")
    except Exception:
        pass
    return ""


async def _query_drug_info(message: str, patient_name: str) -> str:
    """药品信息查询：LLM 提取药名 → HIS 查询 → 结构化输出（2 次 LLM）"""
    from medical_agent.adapters.his import HISAdapter
    from medical_agent.providers.llm import get_llm_qa

    # ① LLM 提取药名（1 次调用）
    drug_query = await _extract_drug_name(message)
    if not drug_query:
        return "未识别到药品名称。请指定药品（如：二甲双胍说明书、阿莫西林相互作用）。"

    # ② HIS 查询（通过别名匹配）
    drug_info = await HISAdapter.search_drugs(drug_query)
    drug_detail = drug_info[0] if drug_info else None
    if not drug_detail:
        return f"未找到药品 {drug_query} 的信息。"

    drug_name = drug_detail["name"]

    # ③ 查患者处方 + 检验指标（如有）
    patient_context = ""
    patient_checks = ""
    if patient_name:
        patient_info = await HISAdapter.get_patient_info(patient_name)
        if patient_info:
            allergies = patient_info.get("allergies", [])
            prescriptions = patient_info.get("prescriptions", [])
            all_meds = set()
            for p in prescriptions:
                for d in p.get("drugs", []):
                    all_meds.add(d["name"])
            if all_meds:
                patient_context = f"患者当前用药：{'、'.join(all_meds)}。"
            if allergies:
                patient_context += f" 过敏史：{'、'.join(allergies)}。"

            # 过敏检查
            med_name_short = drug_detail["name"]
            for alias in drug_detail.get("aliases", []):
                med_name_short = alias  # 用最短别名匹配
            for allergy in allergies:
                if allergy in med_name_short or allergy in drug_detail["name"]:
                    patient_checks = f"\n  ⚠ 警告：患者有{allergy}过敏史，{drug_detail['name']}可能含有{allergy}！"
                    break

            # 相互作用检查（如有多个药物）
            if len(all_meds) >= 1:
                try:
                    from medical_agent.engines.graph.graph_rag import GraphRAGEngine
                    rag = GraphRAGEngine()
                    for med in all_meds:
                        if med != drug_detail["name"]:
                            result = await rag.check_drug_interaction(drug_detail["name"], med)
                            if result and "无" not in str(result):
                                evidence = "指南/明确" if any(k in str(result) for k in ("明确","显著","禁忌","CYP")) else "文献/潜在"
                                patient_checks += f"\n  相互作用: {drug_detail['name']} + {med}（{evidence}）: {result}"
                except Exception:
                    pass

            # ⑤ 查检验指标（药物-检验联动，取每个指标最近一次测量值）
            try:
                from medical_agent.adapters.lis import LISAdapter
                lab_data = await LISAdapter.get_patient_lis_report(patient_name=patient_name)
                if lab_data and lab_data.get("reports"):
                    TARGET = ("肌酐", "血糖", "超敏C反应蛋白")
                    latest_by_name = {}
                    for r in sorted(lab_data["reports"], key=lambda x: x.get("report_date", "")):
                        for ind in r.get("indicators", []):
                            if ind["name"] in TARGET:
                                latest_by_name[ind["name"]] = ind
                    for ind in latest_by_name.values():
                        patient_checks += f"\n  检验: {ind['name']} {ind['value']}{ind['unit']} (参考{ind['ref_range']}) [{ind['status']}]"
            except Exception:
                pass

    # ④ LLM 1 次生成结构化输出
    if patient_context:
        prompt = f"""你是临床药学助手。仅根据以下数据评估该患者是否可以使用此药，不要添加推断。

药品：{drug_detail['name']}
规格：{drug_detail.get('spec','')}

患者：{patient_name}
{patient_context}
{patient_checks}

按以下层级输出（不超过250字，括号内容为格式说明）：
{drug_detail['name']} — {{可以使用✓/慎用⚠/禁用✗}}

患者：{patient_name}
当前用药：{{列出}}
过敏史：{{列出}}

评估详情：
  · 过敏史：{{结论}}
  · 相互作用：{{结论}}
  · 肾功能（如有检验数据）：{{结论}}

院内信息：{drug_detail.get('spec','')} | {drug_detail.get('manufacturer','')} | 库存{'-'.join(str(drug_detail.get('stock','')))}

免责声明：AI建议仅供参考，请以临床诊断和药师审核为准。"""
    else:
        prompt = f"""仅根据以下数据回答药品查询，不要添加推断。

药品：{drug_detail['name']}
规格：{drug_detail.get('spec','')}
厂商：{drug_detail.get('manufacturer','')}
院内库存：{drug_detail.get('stock','')}盒  价格：¥{drug_detail.get('price','')}

按以下格式输出（不超过200字）：
药品查询：{drug_detail['name']}

【院内信息】规格：{drug_detail.get('spec','')}，厂商：{drug_detail.get('manufacturer','')}，库存：{drug_detail.get('stock','')}盒，价格：¥{drug_detail.get('price','')}

【用药提醒】{{1-2句通用用药提醒}}
免责声明：AI建议仅供参考，请以药品说明书和药师指导为准。"""
    llm = get_llm_qa()
    try:
        resp = llm.invoke(prompt)
        return resp.content.strip()
    except Exception:
        result = (
            f"药品查询：{drug_detail['name']}\n\n"
            f"【院内信息】规格：{drug_detail.get('spec','')}，厂商：{drug_detail.get('manufacturer','')}，"
            f"库存：{drug_detail.get('stock','')}盒，价格：¥{drug_detail.get('price','')}"
        )
        if patient_context:
            result += f"\n{patient_context}"
        if patient_checks:
            result += f"{patient_checks}"
        result += "\n免责声明：AI建议仅供参考，请以药品说明书和药师指导为准。"
        return result


# ── 主管编排入口 ──────────────────────────────────────────────────────────

_router = IntentRouter()


async def orchestrate(message: str, user_id: str = "", session_id: str = "", role: str = "patient") -> str:
    """确定性编排：意图分类 → 调用 Worker → 返回结果

    无 ReAct 循环，无 tool-calling 开销。整个编排只有 1 次 LLM 调用（意图分类）。
    """
    # ① 意图分类（1 次 LLM 或规则）
    import time; _t0 = time.time()
    try:
        intent_result = await _router.classify(message)
        intent = intent_result.intent.value
    except Exception:
        intent = "inquiry"  # 失败默认问诊

    # ② 急诊检测
    is_emergency, _ = _router.detect_emergency(message)
    logger.info(f"[编排] 意图={intent}, 急诊={is_emergency}, 耗时={int((time.time()-_t0)*1000)}ms")

    # ③ 操作限制
    if intent == "operation" and role != "admin":
        return "运营数据查询仅限管理员使用。"
    if intent == "drug":
        if role == "patient":
            return "药物咨询仅限医生和药师使用，患者请咨询医师后用药。"
        return await _call_worker(intent, message, session_id, role=role)

    # ④ 调用 Worker
    response = await _call_worker(intent, message, session_id=session_id)

    # ⑤ 急诊前缀
    if is_emergency:
        response = _EMERGENCY_PREFIX + response

    return response
