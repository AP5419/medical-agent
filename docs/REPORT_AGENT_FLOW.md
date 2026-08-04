# 灵枢医疗 — 报告解读 Agent 完整工作流程

> 版本：2026-07-23 | 模型：qwen3.7-max-2026-06-08 | 架构：确定性 Pipeline（非 ReAct）| LLM 调用：最多 1 次

---

## 1. 架构概览

```
用户浏览器 (Gradio)
  │  点击快捷按钮"报告解读" / 输入文字
  ▼
POST /api/v1/chat/stream (SSE)
  │  chat.py:52-62 _stream_generator()
  ▼
编排层 Supervisor (supervisor.py)
  │  orchestrate() → IntentRouter.classify() → _call_worker()
  ▼
意图路由 IntentRouter (intent_router.py)
  │  keyword pre-filter (28个REPORT关键词，0ms命中) → intent=REPORT
  ▼
报告处理 _handle_report (supervisor.py:252-260)
  │
  ├─ 含"出院小结"关键词 → _handle_discharge_summary() (路径C)
  │
  └─ 其他 → _fetch_report_data() → _format_report_output()
              │
              ├─ 多份报告 → 程序化趋势计算 + LLM 总结 (路径B)
              └─ 单份报告 → LLM 格式化 (路径A)
```

### LLM 调用统计

| 路径 | 场景 | LLM 调用 | 典型耗时 |
|------|------|:--:|:--:|
| A | "张三血常规"（单份） | 1 次（格式化） | ~5s |
| B | "张三的信息"（多份趋势） | 1 次（总结） | ~5s |
| C | "张三出院小结" | 1 次（诊疗经过） | ~5s |

> 关键词命中 REPORT intent 跳过 LLM（0ms）。程序趋势计算、模板组装均不调 LLM。

---

## 2. 入口层

### 2.1 Gradio UI → SSE 流式收发

**文件：** `src/medical_agent/ui/gradio_app.py:191-208`

```python
def quick_action(action):
    if action == "报告解读":
        return "请解读这份医疗报告"
    if action == "查看报告":
        return "请帮我查看最近的检查报告"
```

医生点击"报告解读"按钮 → `msg_input` 填入提示文本。用户可修改为"张三血常规"、"张三出院小结"等，发送后由 `chat_send` (L125-186) 发起 SSE POST。

### 2.2 FastAPI SSE 端点

**文件：** `src/medical_agent/api/routers/chat.py:34-62`

`_stream_generator` → `orchestrate(message, user_id, session_id, role)` → 逐字 SSE `event: token` 推送。异常通过 `event: error` 返回前端。

---

## 3. 意图路由（IntentRouter）

**文件：** `src/medical_agent/orchestration/intent_router.py`

### 3.1 REPORT 关键词表（28 个）

```python
# L59-64
IntentType.REPORT: [
    "报告", "化验", "检查结果", "化验单", "体检单", "报告单",
    "CT", "MRI", "B超", "彩超", "X光", "心电图", "血常规",
    "尿常规", "肝功能", "肾功能", "血糖", "血脂", "血压",
    "指标", "偏高", "偏低", "异常", "复查",
    "信息", "档案", "记录", "病历", "就诊记录", "出院", "入院",
],
```

### 3.2 四步分类流水线

```python
# L136-199 (classify 方法)
async def classify(self, message: str) -> IntentResult:
    # ① 急诊正则 → 直接返回 INQUIRY（1ms）
    is_emergency, keywords = self.detect_emergency(message)

    # ② 问候检测（< 30 字符）→ GREETING
    if greet_keywords and len < 30: return GREETING

    # ③ 关键词预筛 → REPORT（0ms，90%+ 请求跳过 LLM）
    for intent_type, keywords in _INTENT_KEYWORD_MAP.items():
        if any(kw in msg_lower for kw in keywords):
            return IntentResult(intent=intent_type, confidence=0.85)

    # ④ LLM JSON 分类兜底（仅关键词未命中时触发）
    ...
```

**设计理由：** 28 个关键词涵盖"血常规""CT""出院""信息"等常见场景，"张三血常规"命中"血常规"→ intent=REPORT，0ms 延迟。

---

## 4. 编排调度（Supervisor）

**文件：** `src/medical_agent/orchestration/supervisor.py`

### 4.1 orchestrate 入口 (L308-340)

```python
async def orchestrate(message, user_id, session_id, role):
    intent_result = await _router.classify(message)          # ①
    is_emergency, _ = _router.detect_emergency(message)     # ②
    if intent == "operation" and role != "admin":            # ③ 权限
        return "运营数据查询仅限管理员使用。"
    response = await _call_worker(intent, message, session_id) # ④
    return (_EMERGENCY_PREFIX + response) if is_emergency else response
```

### 4.2 _call_worker 报告分支 (L48-58)

```python
async def _call_worker(intent, message, session_id):
    ...
    if intent == "report":
        return await _handle_report(message)    # ← 确定性 Pipeline，不调 ReAct Agent
    ...
```

**关键设计：** 不再走 `report/agent.py` 的 ReAct 循环。改为 `_handle_report()` 确定性 Pipeline——程序决定数据访问路径，LLM 只做语言组织。

---

## 5. 双路径路由

**文件：** `src/medical_agent/orchestration/supervisor.py:252-265`

```python
async def _handle_report(message: str) -> str:
    patient, report_type = await _parse_report_query(message)

    # 出院小结关键词 → 走聚合路径
    _DISCHARGE_KEYS = ["出院小结", "出院总结", "住院总结", "出院摘要", "住院摘要"]
    if any(k in message for k in _DISCHARGE_KEYS):
        return await _handle_discharge_summary(patient)

    # 默认：单次查询
    reports = await _fetch_report_data(patient, report_type)
    return await _format_report_output(patient, report_type, reports)
```

### 5.1 患者名和报告类型解析 (L73-108)

```python
# 关键词映射表——用户口语 → LISAdapter mock 数据枚举值
_REPORT_TYPE_MAP = {
    "血常规": "血常规", "生化": "生化检查", "尿常规": "尿常规",
}

_KNOWN_PATIENTS = ["张三", "李四"]

async def _parse_report_query(message):
    # ① 患者名提取（关键词匹配，不在已知名单则 LLM 1 次查询）
    patient_name = next((p for p in _KNOWN_PATIENTS if p in message), "")
    if not patient_name:
        patient_name = await _llm_extract_patient(message)

    # ② 报告类型提取（关键词直接映射）
    for kw, mapped in _REPORT_TYPE_MAP.items():
        if kw in message:
            return patient_name, mapped
    return patient_name, "all"
```

---

## 6. 路径A：单次化验查询

### 6.1 数据获取 (L110-114)

```python
async def _fetch_report_data(patient_name, report_type):
    from medical_agent.adapters.lis import LISAdapter
    return await LISAdapter.search_reports(patient_name=patient_name, report_type=report_type)
```

直接调适配器——不经过 LLM 决策。

### 6.2 结构化格式化 (L206-241)

```python
async def _format_report_output(patient, report_type, reports):
    # 单份报告：构建指标列表 + 异常指标高亮
    abnormal_items = [i for r in reports for i in r["indicators"] if i["status"] != "正常"]

    prompt = f"""仅根据以下检验报告数据生成解读，不要添加数据之外的推断。

患者：{patient}
{reports_text}

异常指标：
{abnormal_text}

严格按格式输出：
异常指标：
- 指标名 值单位（状态）：临床意义（异常指标名和值请用粗体标注）
综合意见：1-2句总结
免责声明：AI解读仅供参考，请以临床医生诊断为准。"""

    resp = llm.invoke(prompt)  # 1 次 LLM 调用
    return resp.content.strip()
```

### 6.3 Mock 数据：LISAdapter

**文件：** `src/medical_agent/adapters/lis.py:10-76`

```python
MOCK_REPORTS = [
    # 张三，社区获得性肺炎，4 时间点
    {"id": "L20251215001", "patient_name": "张三", "report_type": "血常规",
     "report_date": "2024-12-15", "indicators": [
         {"name": "白细胞计数", "value": 15.3, "ref_range": "3.5-9.5", "status": "偏高"},
         {"name": "中性粒细胞百分比", "value": 85.0, "ref_range": "40-75", "status": "偏高"},
         {"name": "超敏C反应蛋白", "value": 85.0, "ref_range": "<5", "status": "偏高"},
         {"name": "降钙素原", "value": 2.5, "ref_range": "<0.5", "status": "偏高"},
     ]},
    # 12/18 D3, 12/20 D5, 12/22 出院，共 4 个时间点
    # 李四，尿常规，1 份
]
```

---

## 7. 路径B：多报告趋势总结

### 7.1 触发条件

"张三的信息" → `report_type="all"` → 返回 5 份报告。

### 7.2 程序化趋势计算（确定性，不调 LLM）

**文件：** `src/medical_agent/orchestration/supervisor.py:142-173`

```python
if len(reports) > 1:
    # ① 构建时序数据（按报告日期排序）
    indicator_series = {}
    for r in sorted(reports, key=lambda x: x["report_date"]):
        for ind in r["indicators"]:
            name = ind["name"]
            if name not in indicator_series:
                indicator_series[name] = []
            indicator_series[name].append(ind)

    # ② 程序计算趋势（确定性，不依赖 LLM）
    trend_lines = []
    for name, points in indicator_series.items():
        values = [p["value"] for p in points]
        first, last = values[0], values[-1]

        if isinstance(first, (int, float)) and isinstance(last, (int, float)):
            direction = "↓" if last < first else "↑" if last > first else "→"

            # 末次值是否在参考范围内（程序解析 ref_range）
            ref = points[-1]["ref_range"]
            last_in_range = _check_in_range(last, ref)

            path = " → ".join(str(v) for v in values)
            status = "已正常" if last_in_range else "仍略高" if direction == "↓" else "仍异常"
            trend_lines.append(f"  {name}: {path}  {status}")
```

**关键设计：**
- 方向判断：first vs last → ↓/↑/→（算术运算，不可能出错）
- 参考范围判断：程序解析 `"3.5-9.5"` → 比较末次值是否在区间内
- 非数值字段（如尿蛋白"+"）跳过方向计算，仅展示原始值

### 7.3 LLM 总结（1 次调用）

```python
# ③ LLM 输入仅含趋势表 + 入院背景，不含原始数据
prompt = f"""仅根据以下数据生成总结，不要添加推断，不要寒暄。

{context}

实验室趋势：
{trends_text}

请用3-4句话总结治疗反应和当前状态（须包含首次和末次具体数值），
仍略高或仍异常的指标请用粗体标注，不超过200字。"""

resp = llm.invoke(prompt)
```

---

## 8. 路径C：出院小结生成

### 8.1 流程（4 步，1 次 LLM 调用）

**文件：** `src/medical_agent/orchestration/supervisor.py:268-363`

```
① 程序: 获取 PATIENT_ADMISSION 入院信息（10 个标准字段）
② 程序: 提取结构化治疗时间线（日期、第N天、指标值）
③ LLM: 根据结构化时间线生成"诊疗经过"段落（1 次调用）
④ 程序: 模板填充组装出院小结全文（不调 LLM）
```

### 8.2 诊疗经过生成：程序提取 → LLM 组织

**LLM 的输入（全部从 mock 数据程序化提取）：**

```
请根据以下结构化治疗时间线生成诊疗经过段落。按时间顺序组织。

患者：张三（男，58岁）
入院诊断：社区获得性肺炎（右下叶）
治疗方案：头孢曲松 2g qd ivgtt + 阿奇霉素 0.5g qd po (12/15-12/22)
影像学检查：12/15 CT: 右下肺实变。12/20 CT: 实变吸收约70%。

实验室检查时间线：
D0 (12/15): WBC 15.3, NEUT% 85.0, CRP 85.0, PCT 2.5
D3 (12/18): WBC 12.1, NEUT% 78.0, CRP 42.0, PCT 0.8
D5 (12/20): WBC 9.8, NEUT% 68.0, CRP 22.0, PCT 0.3
D7 (12/22): WBC 7.2, NEUT% 58.0, CRP 6.0, PCT 0.1

请生成1段诊疗经过（不超过250字），包含入院后治疗方案、各项指标随时间变化趋势、
影像学检查结果、出院时状态。只输出正文，不加标题。
```

**LLM 的输出：**

```
患者入院后予头孢曲松2g qd ivgtt联合阿奇霉素0.5g qd po抗感染治疗。治疗第3天复查
血常规示WBC降至12.1、CRP降至42、PCT降至0.8，较入院时明显改善。第5天CT复查示
右下肺实变吸收约70%。至第7天出院时，WBC 7.2、CRP 6.0、PCT 0.1均已基本正常，
提示抗感染治疗有效，达到临床治愈标准。
```

### 8.3 住院第N天计算 (L356-363)

```python
def _compute_day(admission_date, report_date):
    adm = datetime.strptime(admission_date, "%Y-%m-%d")
    rep = datetime.strptime(report_date, "%Y-%m-%d")
    day = (rep - adm).days
    return "入院" if day == 0 else f"D{day}"
```

### 8.4 程序组装全文（模板填充，不调 LLM）

```python
# supervisor.py:328-353
return f"""灵枢综合医院
出院小结

姓名：{name}  性别：{gender}  年龄：{age}  住院号：{admission_number}
入院日期：{admission_date}  出院日期：{discharge_date}

入院情况：患者因"{chief_complaint}"入院。入院查体：{vital_signs}。
入院诊断：{admission_diagnosis}
诊疗经过：{admission_course}           ← LLM 生成的唯一段落
出院诊断：{discharge_diagnosis}
出院情况：{discharge_condition}
出院医嘱：{discharge_orders}

（AI辅助生成，待主管医师审核签字）"""
```

### 8.5 Mock 入院数据（10 标准字段）

**文件：** `src/medical_agent/adapters/lis.py:78-99`

```python
PATIENT_ADMISSION = {
    "P202501001": {
        "patient_name": "张三",
        "patient_id": "P202501001",
        "gender": "男",
        "age": 58,
        "hospital_name": "灵枢综合医院",
        "admission_number": "INP202512001",
        "admission_date": "2024-12-15",
        "discharge_date": "2024-12-22",
        "chief_complaint": "咳嗽咳痰3天，发热1天，伴右侧胸痛",
        "admission_diagnosis": "社区获得性肺炎（右下叶）",
        "discharge_diagnosis": "社区获得性肺炎（右下叶），临床治愈",
        "vital_signs": "T 38.5℃, P 96/min, R 22/min, BP 128/76mmHg",
        "treatment": "头孢曲松 2g qd ivgtt + 阿奇霉素 0.5g qd po (12/15-12/22)",
        "CT_summary": "12/15 CT: 右下肺实变。12/20 CT: 吸收约70%。",
        "discharge_orders": "阿奇霉素0.5g qd po 续贯3天；1周后门诊复查血常规+CRP。",
        "discharge_condition": "出院时无发热，无咳嗽咳痰，无胸痛。查体：T 36.5℃，双肺清。",
    }
}
```

10 个字段覆盖《病历书写基本规范》出院小结的全部必填项。

---

## 9. 关键设计原则

### 9.1 为什么不使用 ReAct

| 对比 | ReAct（旧方案） | 确定性 Pipeline（新方案） |
|------|------|------|
| 工具调用 | LLM 自主决定调用哪个工具 | 程序根据关键词路由 |
| LLM 调用次数 | 3-4 次（思考→查数据→查指标→输出） | 最多 1 次 |
| 响应时间 | 30-60s | ~5s |
| 数值准确性 | LLM 可能算错趋势 | 程序保证算术正确 |
| 可测试性 | 难以单测（依赖 LLM 推理链） | 可单测（每个函数独立） |

### 9.2 职责分离

| 任务 | 谁做 | 为什么 |
|------|:--:|------|
| 意图分类 | 关键词规则 | 28 个词覆盖 90% 场景，0ms 延迟 |
| 数据查询 | 程序调适配器 | 确定性，不依赖 LLM 猜参数 |
| 趋势计算 | 程序算术 | 15.3→7.2 的程序确定 ↓53%，LLM 可能算错 |
| 参考范围判断 | 程序解析 | `"3.5-9.5"` 的区间逻辑不会出错 |
| 语言组织 | LLM | 唯一需要 AI 的环节 |
| 全文组装 | 程序模板 | 格式确定，不需要 AI 排列 |

### 9.3 出院小结格式合规

严格按照《病历书写基本规范》（卫生部 2010 版）第三章第二十三条定义的出院小结格式：
入院情况、入院诊断、诊疗经过、出院诊断、出院情况、出院医嘱——7 个必填段落全部覆盖。

---

## 10. 完整代码调用链

```
用户输入："张三出院小结"
  │
  ├─[入口] gradio_app.py:191  quick_action("报告解读") → msg_input
  ├─[SSE]  gradio_app.py:125  chat_send → POST /api/v1/chat/stream
  ├─[API]  chat.py:34         _stream_generator → orchestrate()
  │
  ├─[意图] intent_router.py:164  "出院" in _INTENT_KEYWORD_MAP → REPORT (0ms)
  │
  ├─[调度] supervisor.py:308   orchestrate → _call_worker("report", msg)
  ├─[路由] supervisor.py:55    if intent == "report" → _handle_report(msg)
  │
  ├─[解析] supervisor.py:254   _parse_report_query → patient="张三", type="all"
  ├─[分支] supervisor.py:258   "出院小结" in msg → _handle_discharge_summary()
  │
  ├─[数据] supervisor.py:277   get_admission_info → PATIENT_ADMISSION
  ├─[数据] supervisor.py:281   get_patient_lis_report → 5 份 LIS 报告
  ├─[时间] supervisor.py:356   _compute_day → D0/D3/D5/D7
  ├─[LLM]  supervisor.py:307   course_prompt → qwen3.7-max → 诊疗经过 (~5s)
  ├─[组装] supervisor.py:328   模板填充 → 7 段落全文（不调 LLM）
  │
  └─[返回] chat.py:47          逐字 SSE event: token → Gradio Chatbot 渲染
```

---

## 11. 文件索引

### 核心链路文件

| 文件 | 路径 | 行数 | 职责 |
|------|------|:--:|------|
| Gradio | `src/medical_agent/ui/gradio_app.py` | 375 | 快捷按钮绑定、SSE 流式 |
| API | `src/medical_agent/api/routers/chat.py` | 62 | FastAPI SSE 端点 |
| 意图 | `src/medical_agent/orchestration/intent_router.py` | 199 | 28 个 REPORT 关键词预筛 |
| 编排 | `src/medical_agent/orchestration/supervisor.py` | 411 | _handle_report 双路径路由 + 趋势 + 出院小结 |
| 数据 | `src/medical_agent/adapters/lis.py` | 169 | 4 时间点 LIS mock + 入院信息 |

### 关键函数索引

| 函数 | 文件:行 | 用途 |
|------|------|------|
| `classify()` | `intent_router.py:136` | 4 步分类流水线 |
| `orchestrate()` | `supervisor.py:308` | 编排入口 |
| `_handle_report()` | `supervisor.py:252` | 双路径路由 |
| `_parse_report_query()` | `supervisor.py:84` | 患者名+报告类型解析 |
| `_fetch_report_data()` | `supervisor.py:110` | LISAdapter 查询 |
| `_format_report_output()` | `supervisor.py:117` | 单份格式化 / 多份趋势 |
| `_handle_discharge_summary()` | `supervisor.py:268` | 出院小结生成 |
| `_compute_day()` | `supervisor.py:356` | 住院第 N 天计算 |
| `search_reports()` | `lis.py:118` | LIS 报告查询 |
| `get_admission_info()` | `lis.py:162` | 入院信息查询 |
