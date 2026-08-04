# 报告解读 Agent — 核心难点与优化过程

> 基于本项目实际开发中遇到的 5 个可验证工程难题。不涉及与 Drug/Inquiry Agent 的共性难点。

---

## 1. 概述

报告 Agent 与通用 LLM（ChatGPT）的 3 个核心差异：

| 差异 | 通用 LLM | 报告 Agent |
|------|------|------|
| **数据维度** | 用户粘贴的文本或图片 | LIS 时序数据（多时间点）+ HIS 入院信息 + PACS CT 摘要 |
| **格式要求** | 无约束 | 出院小结需符合《病历书写基本规范》7 段必填格式 |
| **输出准确度** | 概率输出 | 15.3→7.2 的下降趋势必须正确，Cr=82 在参考范围内必须正确——不能由 LLM 推算 |

5 个难点均源于"数据聚合 + 格式合规 + 趋势计算"三个维度。

---

## 2. 难点 1：出院小结国标合规

### 问题

出院小结不是"让 LLM 写一篇总结"能解决的问题。必须是《病历书写基本规范》（卫生部 2010 年版，现行有效）定义的 7 段标准格式。

### 关键决策：哪些字段 LLM 生成，哪些程序填充？

| 字段 | 由谁生成 | 原因 |
|------|:--:|------|
| 入院情况（主诉+查体） | 程序填充 | 来自 EMR 结构化字段（`chief_complaint` + `vital_signs`），不需要 AI |
| 入院诊断 | 程序填充 | 来自 `admission_diagnosis` |
| **诊疗经过** | **LLM 生成** | 需要从 LIS 时间线数据中按时间顺序组织成医学叙述——这是唯一需要 AI 的段落 |
| 出院诊断 | 程序填充 | 来自 `discharge_diagnosis` |
| 出院情况 | 程序填充 | 来自 `discharge_condition`（出院时体征描述，医生已写） |
| 出院医嘱 | 程序填充 | 来自 `discharge_orders` |
| 医师签名 | 留空 | 标注"AI辅助生成，待医师审核签字" |

### 诊疗经过的实现

```python
# supervisor.py:268-363 — 4 步：提取时间线 → LLM 诊疗经过 → 模板组装

# ① 从 LIS 提取结构化时间线（程序，不调 LLM）
milestones = []
for r in sorted(reports, key=lambda r: r["report_date"]):
    labs = {ind["name"]: f"{ind['value']}{ind['unit']}" for ind in r["indicators"]}
    milestones.append({
        "day": _compute_day(admission_date, r["report_date"]),  # D0/D3/D5/D7
        "labs": labs,
    })

# ② LLM 仅生成诊疗经过段落（1 次调用）
course_prompt = f"""请根据以下结构化治疗时间线生成诊疗经过。按时间顺序组织，不编造不存在的事件。
治疗方案：{admission['treatment']}
影像学检查：{admission['CT_summary']}
实验室检查时间线：
D0 (12/15): WBC 15.3, CRP 85, PCT 2.5 | CT: 右下肺实变
D3 (12/18): WBC 12.1, CRP 42, PCT 0.8
D5 (12/20): WBC 9.8, CRP 22, PCT 0.3 | CT: 实变吸收约70%
D7 (12/22): WBC 7.2, CRP 6.0, PCT 0.1
只输出正文，不加标题。"""

admission_course = llm.invoke(course_prompt).content.strip()

# ③ 程序组装全文（不调 LLM）
return f"""
灵枢综合医院
出院小结

姓名：张三  性别：男  年龄：58  住院号：INP202512001
入院日期：2024-12-15  出院日期：2024-12-22

入院情况：因"咳嗽咳痰3天，发热1天"入院。入院查体：T 38.5℃...
入院诊断：社区获得性肺炎（右下叶）
诊疗经过：{admission_course}  ← 唯一 LLM 生成的段落
出院诊断：社区获得性肺炎（右下叶），临床治愈
出院情况：出院时无发热，无咳嗽咳痰...
出院医嘱：阿奇霉素0.5g qd po 续贯3天...

（AI辅助生成，待主管医师审核签字）
"""
```

### 设计原则

**不让 LLM 做它不擅长的事。** 时间线提取（哪天的什么指标）→ 程序做，LLM 不做。7 段组装（格式+顺序）→ 程序做，LLM 不做。只有"将结构化时间线转化为连贯医学叙述"——LLM 做。

### 效果

| | ChatGPT | 出院小结 |
|------|------|------|
| 格式 | 自由文本 | 7 段标准格式 |
| 字段来源 | LLM 自行编造 | 全部来自 mock 数据字段 |
| 可审核性 | 无法逐字段核对 | 每段可追溯到数据源 |
| 医师签字 | 无 | 明确标注"待审核签字" |

---

## 3. 难点 2：多报告趋势计算——程序 vs LLM

### 问题

"张三的信息"返回 5 份检验报告。需要计算趋势（WBC 15.3→7.2 是下降吗？7.2 正常吗？）。

让 LLM 做算术 → 可能对也可能错。让程序做算术 → 绝对正确。但程序不会写自然语言。

### 职责分离决策

| 任务 | 谁做 | 原因 |
|------|:--:|------|
| WBC 15.3→7.2 = ↓ | 程序 | 算术运算，确定性 |
| 7.2 ∈ [3.5,9.5] → 正常 | 程序 | 区间判断，确定性 |
| "WBC 15.3→12.1→9.8→7.2 已正常" → 这句话 | LLM | 自然语言组织 |
| "治疗有效，达到临床治愈标准" → 这个判断 | LLM | 需要临床上下文的推理 |

### 实现

```python
# supervisor.py:142-173 — 程序计算趋势 + LLM 总结

# ① 程序计算趋势（确定性）
indicator_series = {}
for r in sorted(reports, key=lambda x: x["report_date"]):
    for ind in r["indicators"]:
        indicator_series.setdefault(ind["name"], []).append(ind)

for name, points in indicator_series.items():
    values = [p["value"] for p in points]
    first, last = values[0], values[-1]

    if isinstance(first, (int, float)):
        direction = "↓" if last < first else "↑" if last > first else "→"

        # 程序解析参考范围判断末次值是否正常
        ref = points[-1]["ref_range"].replace("<", "").strip()
        lo, hi = ref.split("-")
        last_in_range = float(lo) <= last <= float(hi)

        path = " → ".join(str(v) for v in values)
        status = "已正常" if last_in_range else "仍略高" if direction == "↓" else "仍异常"
        trend_lines.append(f"  {name}: {path}  {status}")

# ② LLM 仅做自然语言总结（1 次调用）
prompt = f"""仅根据以下趋势数据生成总结。
实验室趋势：
{trends_text}
请用3-4句话总结治疗反应和当前状态（须包含首次和末次具体数值），仍略高或仍异常的指标请用粗体标注。"""
```

### 为什么这条路正确

LLM 做 `(7.2-15.3)/15.3 = -53%` → 可能对可能错。程序做 → 一定对。但程序不会判断"治疗有效"——这是 LLM 的领域。

**结果：4 行趋势表 + LLM 1 次总结。47s → 10s。**

---

## 4. 难点 3：LLM 输入压缩

### 问题

"张三的信息"触发 `report_type="all"` → 5 份报告 → 20+ 条指标全量序列化 → ~1500 字符的 prompt → LLM 47s 响应 → 页面空白超时。

### 根因

数据量 × LLM 处理时间不成比例增长。20 条指标对整个 LLM 上下文窗口不算大，但 qwen3.7-max 的推理速度对长输入呈超线性增长。

### 优化：程序压缩 → LLM 只做总结

```python
# 旧（全量塞入）
reports_text = "\n\n".join([
    f"报告ID: {r['id']}\n类型: {r['report_type']}\n日期: {r['report_date']}\n"
    + "\n".join(f"  {i['name']}: {i['value']}{i['unit']} ..." for i in r['indicators'])
    for r in reports
])  # → 1500+ 字符

# 新（程序压缩为趋势行）
if len(reports) > 1:
    # 程序计算趋势 → 4 行趋势表，而非 20 行全量指标
    trend_lines.append(f"  {name}: {path}  {status}")
    # → ~150 字符的压缩趋势输入
```

### 效果

| | 全量塞入 | 程序压缩 |
|------|:--:|:--:|
| prompt 大小 | ~1500 字符 | ~150 字符 |
| LLM 响应 | 47s | ~8s |
| 数值准确性 | LLM 自己算趋势 | 程序保证正确 |

---

## 5. 难点 4：报告类型中文匹配

### 问题

用户说"血"→ 需要查到"血常规"报告类型。用户说"生化"→ 需要查到"生化检查"。用户说"尿"→ 需要查到"尿常规"。自由文本与枚举值之间的映射是个隐藏问题。

### 实现

```python
# supervisor.py:73-78 — 简单 + 可维护
_REPORT_TYPE_MAP = {
    "血常规": "血常规",
    "血象":    "血常规",
    "血":      "血常规",    # "张三血信息" → 血常规
    "生化":    "生化检查",
    "肝功能":  "生化检查",   # 肝功能属于生化检查的一种
    "肾功能":  "生化检查",
    "尿常规":  "尿常规",
    "尿":      "尿常规",
}

# 使用（遍历顺序 = 具体优先于宽泛）
report_type = "all"
for kw, mapped in _REPORT_TYPE_MAP.items():
    if kw in message:
        report_type = mapped     # "血常规"先于"血"匹配
```

### 设计要点

- "血常规"在"血"之前——具体匹配优先于单字匹配
- 未命中 → `"all"` → 走多报告趋势路径
- 与 `_KNOWN_PATIENTS` 的患者名提取分离——不耦合

---

## 6. 难点 5：输出格式工程

### 问题

报告解读有 3 种完全不同的输出场景，需要 3 种不同的 prompt 模板：

| 场景 | 输入 | 输出要求 | prompt 策略 |
|------|------|------|------|
| 单份报告 | 1 份×N 指标 | 异常指标列表 + 综合意见 | 结构化列表模板 |
| 多报告趋势 | 5 份压缩趋势行 | 治疗反应总结 | 程序趋势 + LLM 自然语言 |
| 出院小结 | LIS 时间线+HIS 入院 | 7 段标准格式 | 程序组装 + LLM 诊疗经过 |

### 统一入口实现

```python
# supervisor.py:252-260 — 单一路由，3 条输出路径
async def _handle_report(message):
    patient, report_type = await _parse_report_query(message)

    # 出院小结关键词 → 路径 C
    if any(k in message for k in ["出院小结", "出院总结", "住院总结"]):
        return await _handle_discharge_summary(patient)

    reports = await _fetch_report_data(patient, report_type)
    return await _format_report_output(patient, report_type, reports)
    # 内部自动分路径：
    #   len(reports) > 1 → 趋势路径（程序压缩 + LLM 总结）
    #   len(reports) == 1 → 单份路径（LLM 格式化）
    #   len(reports) == 0 → 友好提示
```

### 三种输出的设计差异

| | 单份报告 | 趋势总结 | 出院小结 |
|------|------|------|------|
| 结论位置 | 异常指标列表前置 | 趋势 + 总结 | 7 段标准格式 |
| LLM 调用 | 1 次 | 1 次 | 1 次（仅诊疗经过） |
| 程序参与 | 无 | 趋势计算 | 全文组装 |
| 字数限制 | 300 字 | 200 字 | 不限 |

---

## 7. 总结

| 难点 | 类别 | 核心解法 | 代码位置 |
|------|:--:|------|------|
| 出院小结合规 | 产品 | LLM 仅生成诊疗经过，其余 6 段程序填充 | `supervisor.py:268-363` |
| 程序 vs LLM 算趋势 | 可靠性 | 方向+参考范围→程序，语言组织→LLM | `supervisor.py:142-173` |
| LLM 输入压缩 | 性能 | 20 条全量→4 行趋势行→150 字→8s | 同上迭代 |
| 报告类型匹配 | 入口 | `_REPORT_TYPE_MAP` 具体优先于宽泛 | `supervisor.py:73-78` |
| 输出格式工程 | 产品 | 3 场景 × 3 prompt 模板，统一路由 | `supervisor.py:252-260` |

**一句话总结：报告 Agent 的难点不在"解读一份报告"——ChatGPT 就能做。难点在"出院小结合规格式+多时间点趋势计算+LLM 输入压缩"——这是把医学知识、时序数据和语言模型组织成可交付产品的工程问题。**
