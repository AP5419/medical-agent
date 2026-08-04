# 灵枢医疗 — 患者问诊 Agent 完整执行流程

> 版本：2026-07-23 | 模型：qwen3.7-max-2026-06-08 | 涉及文件：7 个核心模块，527+ 行

---

## 1. 架构总览

```
用户浏览器 (Gradio UI)
  │  POST /api/v1/chat/stream (SSE)
  ▼
FastAPI 路由层 (chat.py)
  │  _stream_generator()
  ▼
编排层 Supervisor (supervisor.py)
  │  orchestrate() → 意图分类 → Worker 调度
  ▼
意图路由 IntentRouter (intent_router.py)
  │  ① 急诊正则 → ② 问候检测 → ③ 关键词预筛 → ④ LLM 兜底
  ▼
问诊 Agent (agent.py)
  │  run_inquiry() → StateGraph 7 节点执行
  ▼
症状标准化管道 (symptom_normalizer.py)
  │  ① LLM 提取 → ② Neo4j 匹配 → ③ Milvus 语义兜底
  ▼
SSE 逐字流式返回 → Gradio Chatbot 渲染
```

### 单轮问诊 LLM 调用次数（优化后）

| 场景 | LLM 调用 | 耗时（估算） |
|------|:------:|:------:|
| 首轮追问（1个症状） | ② symptom extract + ⑤ follow-up question | ~12s |
| 后续追问（2+症状） | ② + ④ candidates + ⑤ follow-up | ~30s |
| 收敛收尾 | ⑥ conclusion | ~20s |

> 早期轮 `_query_candidates` 已延迟到 `len(normalized) >= 2` 时才调用；意图分类在关键词命中时跳过 LLM。

---

## 2. 阶段一：入口层

### 2.1 Gradio UI → SSE 流式收发

**文件：** `src/medical_agent/ui/gradio_app.py:125-186`

```python
async def chat_send(message: str, history: list):
    """前端发送消息，通过 HTTP SSE 流式接收后端回复"""
    if not current_token:
        yield [(text for no-login), msg]
        return

    # 打开 SSE 流
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{BACKEND_URL}/api/v1/chat/stream",
            json={"user_id": ..., "session_id": ..., "message": message}) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                event = data.get("event")
                if event == "token":
                    response += data["data"]
                    yield [{"role": "user", "content": message},
                           {"role": "assistant", "content": response}]
                elif event == "done":
                    break
                elif event == "error":
                    yield [{"role": "user", "content": message},
                           {"role": "assistant", "content": f"错误: {data['data']}"}]
```

**设计理由：**
- 使用 `httpx.AsyncClient.stream()` 而非 WebSocket，简化部署（无需 WS 协议升级）
- SSE `event: token` 逐字推送 + `event: done` 终结，前端实时渲染打字效果
- `session_id` 由前端生成（UUID），后端不复用——避免多个浏览器 tab 共享对话

### 2.2 FastAPI SSE 端点

**文件：** `src/medical_agent/api/routers/chat.py:34-62`

```python
@router.post("/stream", summary="SSE流式对话")
async def chat_stream(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    return StreamingResponse(
        _stream_generator(req, current_user["role"]),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )

async def _stream_generator(req: ChatRequest, role: str):
    from medical_agent.orchestration.supervisor import orchestrate
    try:
        response = await orchestrate(req.message, req.user_id, req.session_id, role)
    except Exception as e:
        yield json.dumps({"event": "error", "data": f"处理异常: {str(e)}"}, ensure_ascii=False) + "\n"
        yield '{"event": "done"}\n'
        return

    for char in response:
        yield json.dumps({"event": "token", "data": char}, ensure_ascii=False) + "\n"
    yield '{"event": "done"}\n'
```

**设计理由：**
- `StreamingResponse` + `text/event-stream` 是 SSE 标准做法，浏览器原生 `EventSource` 可消费
- `X-Accel-Buffering: no` 防止反向代理缓存导致整段输出后才推送
- 异常捕获在生成器内部而非端点层，确保 `event: error` 能被前端收到而非 HTTP 500 静默失败

---

## 3. 阶段二：意图路由 (IntentRouter)

**文件：** `src/medical_agent/orchestration/intent_router.py`

### 3.1 四步分类流水线

```python
# L136-199 (classify 方法)
async def classify(self, message: str) -> IntentResult:
    # ① 急诊正则匹配 → 直接返回 INQUIRY + is_emergency=True
    is_emergency, keywords = self.detect_emergency(message)

    # ② 短文本问候检测 (< 30 字符)
    if any(kw in msg_lower for kw in _GREETING_KEYWORDS) and msg_len < 30:
        return IntentResult(intent=GREETING, confidence=0.9)

    # ③ 关键词预筛（遍历 drug→report→knowledge→operation→inquiry）
    for intent_type, keywords in _INTENT_KEYWORD_MAP.items():
        if any(kw in msg_lower for kw in keywords):
            return IntentResult(intent=intent_type, confidence=0.85)  # ← 跳过 LLM

    # ④ LLM JSON 分类兜底（仅关键词未命中时触发）
    response = await self._llm.ainvoke(INTENT_CLASSIFY_PROMPT.format(message=message))
    # ...解析 {"intent": "inquiry", "confidence": 0.95}
    return IntentResult(intent=..., confidence=...)
```

### 3.2 关键词映射表

```python
# L50-84
_INTENT_KEYWORD_MAP = {
    IntentType.DRUG: ["药品","药物","用药","副作用","剂量","禁忌",...],
    IntentType.REPORT: ["报告","化验单","CT","血常规","指标","偏高",...],
    IntentType.KNOWLEDGE: ["什么是","是什么病","如何治疗","科普",...],
    IntentType.OPERATION: ["统计","报表","运营","数据","KPI",...],
    IntentType.INQUIRY: ["不舒服","疼","痛","挂号","看什么科","发烧","咳嗽",...],
}
```

**设计理由：**
- 关键词预筛可覆盖 90%+ 的用户消息（大多为 symptom 描述），节省 ~14s LLM 调用
- 遍历顺序为 drug→report→knowledge→operation→inquiry，越具体越优先
- 只有关键词全未命中时才回退 LLM JSON 分类，保证罕见意图不丢失
- `confidence=0.85` 明示规则匹配而非模型推理，下游可据此决定是否二次确认

### 3.3 急诊正则匹配

```python
# L117-134
EMERGENCY_KEYWORDS = [
    "胸痛","呼吸困难","意识不清","昏迷","大出血","中风",
    "心肌梗死","心脏骤停","窒息","休克","中毒","溺水",...
]

def detect_emergency(self, message: str) -> tuple[bool, list]:
    matched = []
    for pattern in _EMERGENCY_PATTERNS:  # 预编译的 re.compile
        if pattern.search(message):
            matched.append(kw)
    return len(matched) > 0, matched
```

**设计理由：** 急诊检测必须零延迟——不能等 LLM。正则匹配 < 1ms，命中后 Supervisor 在响应前追加 `EMERGENCY_PREFIX` 警告文字。

---

## 4. 阶段三：编排调度 (Supervisor)

**文件：** `src/medical_agent/orchestration/supervisor.py:69-97`

```python
async def orchestrate(message: str, user_id: str, session_id: str, role: str) -> str:
    # ① 意图分类
    intent_result = await _router.classify(message)
    intent = intent_result.intent.value

    # ② 急诊检测
    is_emergency, _ = _router.detect_emergency(message)

    # ③ 操作限制（非 admin 不能查运营数据）
    if intent == "operation" and role != "admin":
        return "运营数据查询仅限管理员使用。"

    # ④ 调用 Worker Agent
    response = await _call_worker(intent, message, session_id=session_id)

    # ⑤ 急诊前缀
    if is_emergency:
        response = _EMERGENCY_PREFIX + response

    return response
```

### Worker 调用映射

```python
# L22-61
_AGENT_CACHE = {}  # 单例缓存，避免每次重建 Agent

async def _get_or_create_agent(intent: str):
    if intent == "inquiry":
        return run_inquiry          # 直接返回 async function（非 LangGraph graph）
    elif intent == "report":
        return await get_report_agent()     # LangGraph compiled graph
    elif intent == "drug":
        return await get_drug_agent()
    # ...

async def _call_worker(intent: str, message: str, session_id: str = "") -> str:
    agent_or_fn = await _get_or_create_agent(intent)
    if callable(agent_or_fn):  # inquiry 路径
        return await agent_or_fn(message, session_id=session_id)
    result = await agent_or_fn.ainvoke({"messages": [{"role": "user", "content": message}]})
    # 从 LangGraph 结果中提取文本
    for m in reversed(result.get("messages", [])):
        content = getattr(m, "content", "") or ""
        if content and getattr(m, "type", "") != "human":
            return content
```

**设计理由：**
- `inquiry` agent 返回 bare async function 而非 LangGraph graph，因为其内部已封装 checkpoint 管理
- 其余 agent（report/drug/knowledge/operation）走统一的 `.ainvoke()` 入口
- `_AGENT_CACHE` 全局单例避免每次请求重新编译 StateGraph（编译开销大）
- `session_id` 仅在 inquiry 路径使用（其余 agent 为无状态问答）

---

## 5. 阶段四：问诊图执行 (InquiryAgent StateGraph)

**文件：** `src/medical_agent/agents/inquiry/agent.py`

### 5.1 工作流拓扑

```
                    ┌──→ conclude(emergency) ──→ save_record → END
                    │
load_patient → check_emergency
                    │
                    └──→ extract_symptoms → quick_check
                                              │
                          ┌───────────────────┤
                          │                   │
                    conclude(收敛)          END(追问,等待用户输入)
                          │
                    save_record → END
```

### 5.2 图构建代码

```python
# L412-458
def create_inquiry_agent(checkpointer=None):
    workflow = StateGraph(InquiryState)

    # 6 个节点
    workflow.add_node("load_patient", _load_patient)
    workflow.add_node("check_emergency", _check_emergency)
    workflow.add_node("extract_symptoms", _extract_symptoms)
    workflow.add_node("quick_check", _quick_check)
    workflow.add_node("conclude", _conclude)
    workflow.add_node("save_record", _save_record)

    workflow.set_entry_point("load_patient")
    workflow.add_edge("load_patient", "check_emergency")

    # 紧急路由: emergency → conclude | normal → extract
    workflow.add_conditional_edges("check_emergency", _route_after_check,
        {"conclude": "conclude", "extract": "extract_symptoms"})

    workflow.add_edge("extract_symptoms", "quick_check")

    # 收敛路由: 收敛 → conclude | 未收敛 → END（等待用户）
    workflow.add_conditional_edges("quick_check", _route_after_question,
        {"conclude": "conclude", END: END})

    workflow.add_edge("conclude", "save_record")
    workflow.add_edge("save_record", END)

    return workflow.compile(checkpointer=checkpointer)
```

### 5.3 多轮有状态入口

```python
# L398-409 (单例) + L461-527 (入口)
_inquiry_agent_cache = None

def _get_inquiry_agent():
    """MemorySaver 检查点单例：进程内存保存会话状态"""
    global _inquiry_agent_cache
    if _inquiry_agent_cache is None:
        from langgraph.checkpoint.memory import MemorySaver
        _inquiry_agent_cache = create_inquiry_agent(checkpointer=MemorySaver())
    return _inquiry_agent_cache

async def run_inquiry(message: str, session_id: str = "", ...) -> str:
    agent = _get_inquiry_agent()
    thread_id = f"inquiry:{session_id}"
    config = {"configurable": {"thread_id": thread_id}}

    # 检查是否已有检查点（后续轮）
    existing = await agent.aget_state(config)
    is_first_turn = existing is None or not existing.values

    if is_first_turn:
        initial_state = {
            "messages": [{"role": "user", "content": message}],
            "symptoms": [],
            "normalized_symptoms": [],
            "candidate_diseases": [],
            "phase": "initial",
            "symptom_rounds": 0,
            "negation_count": 0,
            "symptom_detail_count": 0,
            ...
        }
    else:
        # 后续轮仅追加消息，症状从检查点恢复
        initial_state = {"messages": [{"role": "user", "content": message}]}

    result = await agent.ainvoke(initial_state, config=config)

    # 优先返回追问问题
    handoff = result.get("handoff_payload", {})
    if handoff and handoff.get("type") == "follow_up":
        return handoff.get("question", "")
    # 否则返回结论
    return result.get("conclusion", "") or ...
```

**设计理由：**
- `MemorySaver` 而非 Redis 简化部署（生产可换 `AsyncRedisSaver`）
- `thread_id = f"inquiry:{session_id}"` 将 session_id 映射到 LangGraph checkpoint namespace
- 首轮传入完整初始状态，后续轮仅传 `messages`——其余字段（`normalized_symptoms` 等）从检查点自动恢复
- `symptom_rounds` 仅在 `_quick_check` 中自增，避免 `_extract_symptoms` 双重计数

### 5.4 各节点详解

#### _load_patient (L51-54)

```python
def _load_patient(state: InquiryState) -> InquiryState:
    state["phase"] = "patient_loaded"
    return state
```
> 预留给未来 HIS/EMR 集成——从数据库加载就诊历史。当前为占位节点。

#### _check_emergency (L57-76)

```python
def _check_emergency(state: InquiryState) -> InquiryState:
    router = IntentRouter()
    last_msg = state["messages"][-1]
    # 从 dict 或 BaseMessage 提取 content
    if isinstance(last_msg, BaseMessage):
        last_message = last_msg.content
    elif isinstance(last_msg, dict):
        last_message = last_msg.get("content", "")
    else:
        last_message = str(last_msg)

    is_emergency, keywords = router.detect_emergency(last_message)
    if is_emergency:
        state["phase"] = "emergency_detected"
        state["symptoms"] = state.get("symptoms", []) + [
            {"type": "emergency", "keywords": keywords}
        ]
    else:
        state["phase"] = "checked"
    return state
```

#### _extract_symptoms (L79-117)

```python
async def _extract_symptoms(state: InquiryState) -> InquiryState:
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

    # 否定回答过滤：计数连续否定，切换追问策略用
    _NEGATION_SET = {"没有","无","没","不","a","b","c","A","B","C",
                     "是","否","yes","no","不知道","不清楚","还行","还好","none"}
    if last_message and last_message.strip() in _NEGATION_SET:
        state["negation_count"] = state.get("negation_count", 0) + 1
        state["symptom_detail_count"] = state.get("symptom_detail_count", 0)
        state["normalized_symptoms"] = state.get("normalized_symptoms", [])
        state["symptom_detail"] = state.get("symptom_detail") or {}
        state["phase"] = "symptoms_extracted"
        return state  # 跳过标准化，保留已有症状

    # 非否定回答（len>1）：重置否定计数，计为细节
    if last_message and len(last_message.strip()) > 1:
        state["negation_count"] = 0
        state["symptom_detail_count"] = state.get("symptom_detail_count", 0) + 1

    if last_message:
        result = await normalize_symptoms(last_message)
        prev_normalized = state.get("normalized_symptoms", [])
        new_symptoms = result["all_standard"]
        state["normalized_symptoms"] = list(set(prev_normalized + new_symptoms))
        state["symptom_detail"] = {
            "matched": result["matched"],
            "mapped": result["mapped"],
            "unmatched": result["unmatched"],
        }
        state["symptoms"] = state.get("symptoms", []) + result["all_standard"]
    else:
        state["normalized_symptoms"] = []
        state["symptom_detail"] = {"matched": [], "mapped": {}, "unmatched": []}

    state["phase"] = "symptoms_extracted"
    return state
```

> **否定过滤 + 计数设计理由：**
> - "没有"/"A"/"B" 不是症状，不进入标准化管道
> - `negation_count` 追踪连续否定次数：在 `_quick_check` 中用于切换追问策略（伴随→细节→收敛）
> - `symptom_detail_count` 追踪已收集的细节数：用户回答"几天"/"走楼梯加重"等非否定非症状短文本时 +1
> - 非否定回答重置 `negation_count=0`：用户一旦提供有效信息即跳出否定循环

#### _save_record (L301-304)

```python
def _save_record(state: InquiryState) -> InquiryState:
    state["phase"] = "record_saved"
    return state
```
> 预留给问诊记录持久化到 MySQL。

---

## 6. 阶段五：症状标准化（3 层管道）

**文件：** `src/medical_agent/agents/inquiry/symptom_normalizer.py`

### 6.1 管道入口

```python
# L169-220
async def normalize_symptoms(user_input: str) -> dict:
    llm = get_llm_qa()

    # ① LLM 提取 + 标准化
    symptoms = await extract_symptoms_layer1(user_input, llm)
    if not symptoms:
        # LLM 失败兜底: 用原始输入避免症状归零
        fallback = user_input
        if isinstance(user_input, dict):
            fallback = user_input.get("content", str(user_input))
        return {"matched": [], "mapped": {}, "unmatched": [fallback],
                "all_standard": [fallback]}

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

    # 所有症状都纳入 all_standard（包括 still_unmatched）
    all_standard = matched + list(mapped.values()) + still_unmatched

    return {
        "matched": matched,
        "mapped": mapped,
        "unmatched": still_unmatched,
        "all_standard": all_standard,
    }
```

### 6.2 第一层：LLM 口语→术语 (L48-79)

```python
SYMPTOM_EXTRACT_PROMPT = """你是医疗术语标准化专家。
任务：从用户的描述中提取所有症状，并将每个症状转换为标准医学术语。

标准化规则：
- 发烧/烧/低烧/高烧 → 发热
- 肚子疼/肚痛/腹部疼痛/肚子不舒服 → 腹痛
- 拉肚子/跑肚/稀便 → 腹泻
- 浑身没劲/没力气/疲惫 → 乏力
- 恶心想吐/想呕吐/胃部不适 → 恶心
- 头疼/头部疼痛/偏头痛 → 头痛
- 咳嗽/干咳/咳痰 → 咳嗽
...

用户描述：{user_input}
返回JSON: {"symptoms": ["标准化症状1", "标准化症状2"]}"""

async def extract_symptoms_layer1(user_input: str, llm) -> list[str]:
    response = llm.invoke(SYMPTOM_EXTRACT_PROMPT.format(user_input=user_input))
    json_match = re.search(r'\{[^{}]*\}', text)
    data = json.loads(json_match.group())
    return [s.strip() for s in data.get("symptoms", [])]
```

**设计理由：**
- 单次 LLM 调用完成提取+标准化，无需多轮 ask→get 循环
- Prompt 中包含 20+ 条显式映射规则（few-shot），提高准确率
- 返回 `[]` 而非抛异常，上层用兜底处理

### 6.3 第二层：Neo4j 精确匹配 (L82-111)

```python
async def match_neo4j_layer2(symptoms: list[str], neo4j_driver) -> tuple[list, list]:
    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (s:Symptom) WHERE s.name IN $names RETURN s.name AS name",
            names=symptoms,
        )
        records = await result.data()
        matched_set = {r["name"] for r in records}

    matched = [s for s in symptoms if s in matched_set]
    unmatched = [s for s in symptoms if s not in matched_set]
    return matched, unmatched
```

**设计理由：**
- Neo4j 存储标准症状节点（`(:Symptom {name: "腹痛"})`），精确匹配确认术语在图谱中存在
- 单个 Cypher `IN` 查询批量匹配，非逐条 N+1
- Neo4j 失败时全部进 `unmatched`，不中断管道

### 6.4 第三层：Milvus 语义兜底 (L114-166)

```python
SIMILARITY_THRESHOLD = 0.85  # 余弦相似度阈值

async def match_milvus_layer3(unmatched, embedding_model, milvus_alias):
    collection = Collection("symptoms", using=milvus_alias)
    collection.load()

    mapped = {}
    still_unmatched = []
    for symptom in unmatched:
        query_vec = embedding_model.embed_query(symptom)
        results = collection.search(
            data=[query_vec], anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=1, output_fields=["name"],
        )
        if results and results[0] and results[0][0].distance >= SIMILARITY_THRESHOLD:
            mapped[symptom] = results[0][0].entity.get("name", "")
        else:
            still_unmatched.append(symptom)

    return mapped, still_unmatched
```

**设计理由：**
- 仅对 Neo4j 未命中的词做向量检索，避免全量向量计算
- 阈值 0.85 经过调优——太低引入噪音（"腹痛"→"胸痛"），太高丢失合法映射
- **关键设计:** `still_unmatched` 纳入 `all_standard`，确保如"膝关节痛"之类圖庫未覆盖的词不被丢弃

---

## 7. 阶段六：企业级收敛（3 因子评分）

**文件：** `src/medical_agent/agents/inquiry/agent.py:312-376`

### 7.1 收敛评分函数

```python
def _should_conclude(state: InquiryState) -> tuple[bool, str]:
    rounds = state.get("symptom_rounds", 0)
    normalized = state.get("normalized_symptoms", [])
    candidates = state.get("candidate_diseases", [])

    # ① 强制停止：追问 > 5 轮
    if rounds >= 5:
        return True, f"轮次={rounds}>=5 强制停止"

    # ② 置信度收敛（权重 30%）: top1 >= 0.5 且 gap >= 0.15
    top1_prob = candidates[0].get("probability", 0) if candidates else 0
    top2_prob = candidates[1].get("probability", 0) if len(candidates) > 1 else 0
    confidence_ok = top1_prob >= 0.5 and (top1_prob - top2_prob) >= 0.15

    # ③ 症状维度（权重 40%）: >= 3 个标准化症状
    symptoms_ok = len(normalized) >= 3

    # ④ 追问轮次（权重 30%）: >= 2 轮
    rounds_ok = rounds >= 2

    score = symptoms_ok * 0.4 + confidence_ok * 0.3 + rounds_ok * 0.3
    return (score >= 0.6, f"评分{score:.1f}")
```

### 7.2 评分矩阵

| 维度=3 | 置信度=ok | 轮次=2 | 评分 | 结论 |
|:---:|:---:|:---:|:---:|------|
| ✓ | ✓ | ✓ | 1.0 | 收尾 |
| ✓ | ✗ | ✓ | 0.7 | 收尾（维度足够） |
| ✗ | ✓ | ✓ | 0.6 | 收尾（置信度兜底） |
| ✗ | ✗ | ✓ | 0.3 | 继续追问 |
| ✗ | ✗ | ✗ | 0.0 | 首轮，继续 |

### 7.2.1 因子详解

每个因子的数据来源、阈值条件和设计理由：

**维度（权重 40%）** — `len(normalized_symptoms) >= 3`

```
来源: _extract_symptoms (L79-117)
  → normalize_symptoms() 三层管道
  → all_standard = matched + mapped.values() + still_unmatched
  → 累积去重: list(set(prev + new))
  
当前值 = len(normalized_symptoms)  例: ['膝关节痛'] → 1
阈值   = 3                           ≥3 → symptoms_ok = True
贡献   = 0.4 × (1 或 0)

为什么权重最大 (40%):
  LLM 自评的概率不校准（可能给腹痛打 0.85）。
  症状数量是客观事实不会撒谎——维度足够时即使置信度=0 也能收尾 (0.7 分)。
```

**置信度（权重 30%）** — `top1 >= 0.5 且 gap >= 0.15`

```
来源: _query_candidates (L120-167)
  → LLM 根据症状输出 JSON: {"diseases": [{name, probability, ...}]}
  → candidates[0].probability  例: 0.85
  → gap = top1 - top2          例: 0.85 - 0.70 = 0.15

当前值 = top1=0.85, top2=0.70
条件   = top1>=0.5 AND gap>=0.15 → confidence_ok = True
贡献   = 0.3 × (1 或 0)

为什么 gap>=0.15:
  避免 top1=agreement=top2=0.95 时虚假满足——多个疾病概率接近说明不确定性高。

为什么经常是 0.00:
  首轮症状 < 2 时 _quick_check 跳过 _query_candidates（性能优化）。
  candidates=[] → top1=0.00 → 置信度贡献=0。
```

**轮次（权重 30%）** — `symptom_rounds >= 2`

```
来源: _quick_check (L332)
  → state["symptom_rounds"] += 1  仅在 _quick_check 中自增

当前值 = 1
阈值   = >= 2                        → rounds_ok = True
贡献   = 0.3 × (1 或 0)

为什么需要 >= 2:
  首轮症状通常不完全（用户可能只说"肚子痛"）。
  至少追问 1 轮后才考虑收尾，避免信息不充分。
```

**强制停止** — `symptom_rounds >= 5`

```
直接收尾，不参与加权评分。
防止仅 1 个症状且用户持续回答"没有"时的死循环。
```

---

### 7.2.2 实例：膝关节痛，log 行逐行解读

```
[问诊] 收敛评分=0.0 (维度=False(1) 置信度=False(0.00/0.00) 轮次=False(1))

score 计算过程:
  维度   = False(len=1<3)     × 0.4 = 0.0
  置信度 = False(top1=0.00)   × 0.3 = 0.0    ← candidates=[](症状<2,未查询)
  轮次   = False(rounds=1<2)  × 0.3 = 0.0
  ─────────────────────────────────────────────
  score  = 0.0  <  0.6  →  追问"除了膝关节痛,有没有其他伴随症状？"
```

### 7.2.3 收敛路径推演

| 轮次 | 症状 | 维度 | 置信度 | 轮次 | 评分 | 判定 |
|:---:|------|:---:|:---:|:---:|:---:|------|
| 1 | 膝关节痛 | ✗(1) | ✗(0.00) | ✗(1) | **0.0** | 追问 |
| 2 | 膝关节痛 + 无伴随 | ✗(1) | ✗(0.00) | ✓(2) | **0.3** | 追问 |
| 3 | 膝关节痛+肿胀+受限 | ✓(3) | ✗(0.00) | ✓(3) | **0.7** | ✅ 收尾 |

> 仅有 1 个症状且持续否定时，最多撑到第 5 轮（`rounds >= 5` 强制停止）。

### 7.3 quick_check 收敛节点（含自适应追问）

```python
def _quick_check(state: InquiryState) -> InquiryState:
    state["symptom_rounds"] = state.get("symptom_rounds", 0) + 1

    # 仅症状 >= 2 时查询候选疾病（优化：节省 ~20s/早期轮）
    normalized = state.get("normalized_symptoms", [])
    if len(normalized) >= 2:
        state = _query_candidates(state)

    should_stop, reason = _should_conclude(state)
    if should_stop:
        state["phase"] = "ready_to_conclude"
        state["handoff_payload"] = None
        return state

    # ── 自适应追问策略（3 分支） ──
    candidates = state.get("candidate_diseases", [])
    llm = get_llm_qa()
    symptoms_text = "、".join(normalized) if normalized else "未明确"
    negation_count = state.get("negation_count", 0)
    detail_count = state.get("symptom_detail_count", 0)
    rounds = state.get("symptom_rounds", 1)

    if negation_count >= 2 and detail_count < 2 and rounds >= 2:
        # 分支②：连续否定 → 改问症状细节（时长/程度/诱因）
        prompt = (
            f"已知症状：{symptoms_text}。患者已确认无其他伴随症状。"
            f"请针对现有症状追问1个细节问题（如：持续时间、疼痛程度、"
            f"加重/缓解因素、诱发原因等），给出2-3个选项降低输入难度。"
            f"只输出问题本身。"
        )
    elif detail_count >= 2 and negation_count >= 2 and rounds >= 3:
        # 分支③：细节足够 + 确认无伴随 → 提前收敛（不限 5 轮兜底）
        logger.info(f"[问诊] 否定追踪收敛: negation={negation_count}, detail={detail_count}")
        state["phase"] = "ready_to_conclude"
        state["handoff_payload"] = None
        return state
    else:
        # 分支①：默认 → 追问伴随症状
        cand_hint = ""
        if len(candidates) >= 2:
            cand_hint = f"候选: {candidates[0].get('name')} vs {candidates[1].get('name')}。"
        prompt = (
            f"已知症状：{symptoms_text}。" + cand_hint +
            "请用一句话追问1个最关键的问题。优先问患者有没有其他伴随症状，" +
            "给出2-3个选项降低用户输入难度。只输出问题本身。"
        )

    resp = llm.invoke(prompt)
    state["handoff_payload"] = {"type": "follow_up", "question": resp.content}
    return state
```

### 7.3.1 自适应追问状态机

```
用户: "膝关节痛" (negation=0, detail=0)
  ↓
分支① → "除了膝痛，有没有发热乏力等伴随症状？"
  ↓ 用户: "没有" (negation=1, detail=0)
分支① → "有没有红肿或其他关节疼痛？"
  ↓ 用户: "没有" (negation=2, detail=0) → 条件触发
分支② → "疼痛持续多久？A.几小时 B.几天 C.几周"
  ↓ 用户: "几天" (negation=0, detail=1)
分支② → "走楼梯或下蹲时加重吗？A.会 B.不会"
  ↓ 用户: "会" (negation=0, detail=2) → 条件触发
分支③ → convergence → _conclude
  ↓
输出: 膝骨关节炎（可能性极高），推荐科室: 骨科
```

> 关键设计：`negation` 和 `detail` 由 `_extract_symptoms` 维护，`_quick_check` 仅读取决策。
> 非否定有效回答会自动重置 `negation=0`，确保用户一旦提供细节就不再死循环。

### 7.4 概率→中文标签

```python
# L214-224
def _prob_label(p: float) -> str:
    if p >= 0.8: return "可能性极高"
    if p >= 0.6: return "可能性大"
    if p >= 0.4: return "中等可能"
    if p >= 0.2: return "略有可能"
    return "可能性较低"
```

---

## 8. 阶段七：结论生成

**文件：** `src/medical_agent/agents/inquiry/agent.py:227-298`

```python
def _conclude(state: InquiryState) -> InquiryState:
    normalized = state.get("normalized_symptoms", [])
    candidates = state.get("candidate_diseases", [])

    if not candidates and not normalized:
        state["conclusion"] = "根据您提供的信息，暂时无法做出明确判断..."
        state["phase"] = "concluded"
        return state

    symptoms_text = "、".join(normalized) if normalized else "未提供"

    # 结构化候选疾病文本（保留概率标签、科室、理由供 LLM 使用）
    candidate_lines = []
    for d in candidates[:5]:
        candidate_lines.append(
            f"{d['name']}（{_prob_label(d['probability'])}，"
            f"{d['recommended_department']}）：{d['reasoning']}"
        )
    candidates_text = "\n".join(candidate_lines)

    top_dept = candidates[0].get("recommended_department", "内科")

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

    # 严格模板约束 LLM 输出（不要寒暄、不要称呼、固定格式）
    prompt = f"""你是分诊导诊助手。请仅根据以下候选疾病数据生成结论。
不要添加数据之外的推断，不要寒暄、不要称呼。

症状：{symptoms_text}
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

    try:
        response = llm.invoke(prompt)
        state["conclusion"] = response.content.strip()
    except Exception:
        # LLM 失败时直接用结构化数据拼装（纯数据驱动，无 LLM 依赖）
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

    state["messages"] = state.get("messages", []) + [
        {"role": "assistant", "content": state["conclusion"]}
    ]
    state["phase"] = "concluded"
    return state
```

**设计理由：**
- `candidates_text` 从 dict 逐字段拼装而非 `str()`，保留科室、概率、理由的结构化信息
- Prompt 严格要求"不要寒暄 + 固定模板"→ 输出为 Agent 风格而非 ChatGPT 风格
- `_DEPT_WARNINGS` 按科室动态选择紧急提醒，避免"膝盖疼"出现"腹痛"警告
- LLM 失败时程序员直接拼装（无需 LLM），确保任何情况下都有可用输出
- `_prob_label` 替换 `85%` 为"可能性极高"→ 患者可读

---

## 9. 关键数据结构

### 9.1 InquiryState (agent.py:37-48)

```python
class InquiryState(TypedDict):
    messages: Annotated[list, operator.add]    # 对话历史（追加而非覆盖）
    symptoms: list                             # 原始症状
    normalized_symptoms: list                  # 标准化症状名（累积去重）
    symptom_detail: Optional[dict]             # {matched, mapped, unmatched}
    candidate_diseases: list                   # [{"name":..., "probability":..., ...}]
    phase: str                                 # 当前阶段
    patient_context: Optional[dict]            # 患者上下文
    conclusion: Optional[str]                  # 最终结论
    handoff_payload: Optional[dict]            # 追问消息 {"type":"follow_up","question":"..."}
    symptom_rounds: int                        # 症状采集轮次
    negation_count: int                        # 连续否定回答次数
    symptom_detail_count: int                  # 已收集的症状细节数
```

关键设计：
- `messages` 使用 `operator.add` reducer → 多轮追加而非覆盖
- `normalized_symptoms` 无 reducer → 由 `_extract_symptoms` 显式 `list(set(prev + new))` 累积去重
- `handoff_payload` 收敛时显式设为 `None` 避免跨轮污染
- `negation_count` + `symptom_detail_count` → 自适应追问状态机（方案 D）：连续 2 次否定后切换问细节，细节 >= 2 后提前收敛

### 9.2 IntentResult (intent_router.py:21-27)

```python
@dataclass
class IntentResult:
    intent: IntentType              # INQUIRY|REPORT|DRUG|KNOWLEDGE|OPERATION|GREETING
    confidence: float = 0.5         # 置信度
    is_emergency: bool = False      # 是否急诊
    emergency_keywords: list = field(default_factory=list)
```

---

## 10. 完整文件索引

### 核心链路文件

| 文件 | 路径 | 行数 | 职责 |
|------|------|:--:|------|
| Gradio UI | `src/medical_agent/ui/gradio_app.py` | 375 | 前端入口、SSE 流式收发、登录/注册 |
| API 路由 | `src/medical_agent/api/routers/chat.py` | 62 | FastAPI SSE 端点、`ChatRequest` 模型 |
| 编排层 | `src/medical_agent/orchestration/supervisor.py` | 97 | `orchestrate()` 确定性编排、Worker 调度 |
| 意图路由 | `src/medical_agent/orchestration/intent_router.py` | 199 | 4 步分类流水线、急诊检测 |
| 问诊 Agent | `src/medical_agent/agents/inquiry/agent.py` | 527 | StateGraph 7 节点、收敛评分、结论生成 |
| 症状标准化 | `src/medical_agent/agents/inquiry/symptom_normalizer.py` | 220 | 3 层匹配管道（LLM+Neo4j+Milvus） |
| LLM Provider | `src/medical_agent/providers/llm.py` | 74 | `get_llm_qa()` 工厂、模型适配 |

### 基础设施文件

| 文件 | 路径 | 职责 |
|------|------|------|
| 配置 | `src/medical_agent/core/config.py` | 环境变量管理、默认值 |
| 环境 | `.env` | `CHAT_MODEL`、API Key、端口配置 |
| Neo4j | `src/medical_agent/infra/neo4j.py` | Neo4j 异步驱动连接池 |
| Milvus | `src/medical_agent/infra/milvus.py` | Milvus 连接管理 |
| Redis | `src/medical_agent/infra/redis.py` | Redis 连接池（checkpointer 预留） |
| 记忆引擎 | `src/medical_agent/engines/memory/memory.py` | MemorySaver 检查点工厂 |

---

## 附录：一次完整问诊的决策树

```
用户: "肚子不舒服，有点恶心"
  │
  ├─[SSE] chat_stream → orchestrate()
  │
  ├─[意图] classify("肚子不舒服...")
  │   └─ "不舒服" in INQUIRY keywords → IntentType.INQUIRY (0ms, 跳过LLM)
  │
  ├─[调度] _call_worker("inquiry") → run_inquiry(message, session_id)
  │
  ├─[图执行] agent.ainvoke(state, config={thread_id})
  │   │
  │   ├─ load_patient: phase="patient_loaded"
  │   ├─ check_emergency: 无急诊关键词 → phase="checked"
  │   ├─ extract_symptoms:
  │   │   ├─ ① LLM: "肚子不舒服"→"腹痛", "有点恶心"→"恶心"
  │   │   ├─ ② Neo4j: 腹痛✓, 恶心✓ → matched=['腹痛','恶心']
  │   │   ├─ ③ Milvus: unmatched=[] → skip
  │   │   └─ normalized_symptoms = ['腹痛','恶心'], symptom_detail = {...}
  │   │
  │   ├─ quick_check:
  │   │   ├─ symptom_rounds += 1 → 1
  │   │   ├─ len(normalized)=2 → _query_candidates()
  │   │   │   └─ LLM: 急性胃肠炎(85%), 消化性溃疡(70%), ...
  │   │   ├─ _should_conclude():
  │   │   │   ├─ rounds=1<5 ✗
  │   │   │   ├─ confidence: top1=0.85≥0.5 ✓, gap=0.15≥0.15 ✓ → True
  │   │   │   ├─ symptoms: len=2<3 ✗
  │   │   │   ├─ rounds_ok: 1<2 ✗
  │   │   │   └─ score = 0.4*0 + 0.3*1 + 0.3*0 = 0.3 < 0.6 → 不收敛
  │   │   └─ 生成追问: "还有没有腹泻、发热等其他症状？(A.有/B.没有)"
  │   │
  │   └─ _route_after_question: phase="asking"≠"ready_to_conclude" → END
  │
  └─[返回] handoff_payload.question → SSE 逐字推送

用户: "有腹泻"
  │
  ├─ run_inquiry("有腹泻", session_id)
  │   ├─ aget_state(config) → 存在 → is_first_turn=False
  │   ├─ initial_state = {"messages": [{"role":"user","content":"有腹泻"}]}
  │   │   # normalized_symptoms=['腹痛','恶心'] 从 checkpoint 恢复
  │   │
  │   ├─ extract_symptoms:
  │   │   ├─ ① LLM: "有腹泻" → ['腹泻']
  │   │   ├─ ② Neo4j: 腹泻✓ → matched=['腹泻']
  │   │   └─ normalized_symptoms = list(set(['腹痛','恶心'] + ['腹泻']))
  │   │                          = ['腹痛','恶心','腹泻']
  │   │
  │   ├─ quick_check:
  │   │   ├─ symptom_rounds += 1 → 2
  │   │   ├─ len(normalized)=3 → _query_candidates()
  │   │   │   └─ LLM: 急性胃肠炎(90%), 细菌性痢疾(65%), ...
  │   │   └─ _should_conclude():
  │   │       ├─ rounds=2<5 ✗
  │   │       ├─ symptoms: 3≥3 ✓ (40%)
  │   │       ├─ rounds_ok: 2≥2 ✓ (30%)
  │   │       └─ score = 0.4*1 + 0.3*1 + 0.3*1 = 1.0 ≥ 0.6 → 收敛！
  │   │
  │   └─ _route_after_question: phase="ready_to_conclude" → "conclude"
  │
  ├─ _conclude:
  │   └─ 候选: 急性胃肠炎(可能性极高)、消化性溃疡(可能性大)、...
  │   └─ 推荐科室: 消化内科
  │   └─ 输出结构化结论（无 ChatGPT 风格寒暄）
  │
  └─[返回] conclusion → SSE 逐字推送
```
