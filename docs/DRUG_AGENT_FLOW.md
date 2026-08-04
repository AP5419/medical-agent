# 灵枢医疗 — 药物咨询 Agent 完整工作流程

> 版本：2026-07-24 | 模型：qwen3.7-max-2026-06-08 | 架构：确定性 Pipeline（非 ReAct）| LLM 调用：药品查询 2 次，处方审核 1 次

---

## 1. 架构概览

### 角色权限矩阵

| 操作 | 患者 | 医生 | 药师 | admin |
|------|:--:|:--:|:--:|:--:|
| 药品查询 | ✗ | ✅ | ✅ | ✅ |
| 处方审核 | ✗ | ✗ | ✅ | ✗ |
| intent=drug 入口 | ✗ | ✅ | ✅ | ✅ |

### 3 条 Pipeline 路径

```
用户输入
  │
  ├─ patient → "药物咨询仅限医生和药师使用" ─→ 直接拒绝
  │
  ├─ doctor/pharmacist/admin → intent=DRUG → _handle_drug(message, role)
  │     │
  │     ├─ 含"审核/审方/处方核对"关键词 + role=pharmacist
  │     │   → _review_prescription(patient_name)
  │     │     • HIS 查患者处方 + 过敏史
  │     │     • GraphRAG 逐对检查药物相互作用
  │     │     • LLM 1 次生成审核报告
  │     │
  │     └─ 其他
  │         → _query_drug_info(message, patient_name)
  │           • LLM 1 次提取药名（_extract_drug_name）
  │           • HISAdapter 查药品信息（名称+别名匹配）
  │           • 有患者上下文：HIS 查处方+过敏史，GraphRAG 查相互作用
  │           • LLM 1 次格式化输出
  │
  └─ inquiry → run_inquiry（问诊状态机）
     report → _handle_report（报告解读）
```

### LLM 调用统计

| 路径 | 场景 | LLM 调用 | 内容 |
|------|------|:--:|------|
| _query_drug_info | 医生查药（无患者上下文） | 2 次 | ① 提取药名 ② 格式化输出 |
| _query_drug_info | 医生查药（有患者上下文） | 2 次 | ① 提取药名 ② 临床评估 |
| _review_prescription | 药师审核处方 | 1 次 | 生成审核报告 |

### 数据来源（2 层）

| 数据源 | 查什么 | 技术 | 用途 |
|------|------|------|------|
| **HISAdapter** (his.py) | 药品库存/价格、患者处方、过敏史 | mock dict → 生产换 MySQL | 机构特有数据 |
| **GraphRAG + Neo4j** (graph_rag.py) | 药物相互作用（双药→共享疾病→联合用药风险） | Cypher 查询 Neo4j 知识图谱 | 医学知识推理 |
| **LISAdapter** (lis.py) | 患者检验指标（Cr/血糖/CRP） | mock dict → 生产换 HIS API | 药物-检验联动 |

---

## 2. 入口层

### 2.1 意图路由 → 角色拦截（orchestrate）

**文件：** `src/medical_agent/orchestration/supervisor.py:644-651`

```python
# ③ 操作限制
if intent == "operation" and role != "admin":
    return "运营数据查询仅限管理员使用。"
if intent == "drug":
    if role == "patient":
        return "药物咨询仅限医生和药师使用，患者请咨询医师后用药。"
    return await _call_worker(intent, message, session_id, role=role)
```

**设计理由：** 患者侧药物咨询有法律风险——患者可能依据 AI 建议自行用药或停药。医生/药师侧作为辅助决策工具，使用者具备专业判断能力。

### 2.2 Worker 调度 → _handle_drug（_call_worker）

**文件：** `src/medical_agent/orchestration/supervisor.py:48-58`

```python
async def _call_worker(intent: str, message: str, session_id: str = "", role: str = "") -> str:
    agent_or_fn = await _get_or_create_agent(intent)
    if callable(agent_or_fn):  # inquiry
        return await agent_or_fn(message, session_id=session_id)
    if intent == "report":
        return await _handle_report(message)
    if intent == "drug":
        return await _handle_drug(message, role=role)  # ← 带角色参数
```

**关键设计：** 不走 `drug/agent.py` 的 ReAct 循环。改为 `_handle_drug()` 确定性 Pipeline。

---

## 3. 意图路由（IntentRouter）

**文件：** `src/medical_agent/orchestration/intent_router.py`

### 3.1 DRUG 关键词表（15+ 词，L52-57）

```python
IntentType.DRUG: [
    "药品", "药物", "吃药", "用药", "剂量", "用法", "用量",
    "副作用", "禁忌", "说明书", "处方", "抗生素", "消炎药",
    "降压药", "降糖药", "阿司匹林", "头孢", "阿莫西林",
    "服用", "口服", "外用药", "注射", "皮试",
],
```

### 3.2 LLM 增强分类（L3 兜底，L88-105）

```python
INTENT_CLASSIFY_PROMPT = """你是一个医疗意图分类器。从以下6个意图中选择最匹配的一个。

意图类型及区分规则（必须选一个）：
- drug: 药物咨询——用户提及任何药品相关话题："XX可以吃吗""怎么吃""副作用""用法用量"
  注意：消息中提到任何药品名称（中文/英文/商品名）优先选 drug
  ...
```

**设计理由：** 关键词覆盖常见词（"副作用""说明书"），LLM 兜底覆盖"格华止怎么用""那个降糖的药"等变体。两者共存——关键词 0ms 零成本，LLM 仅兜底。

---

## 4. 角色权限控制

**文件：** `src/medical_agent/orchestration/supervisor.py:374-382`

```python
async def _handle_drug(message: str, role: str = "") -> str:
    """药物咨询确定性 pipeline：药师→审核+查询，医生→仅查询"""
    patient_name = _extract_patient_name(message)

    _REVIEW_KEYS = ["审核", "审查", "处方", "核对", "检查处方", "审方"]
    if any(k in message for k in _REVIEW_KEYS) and patient_name:
        if role != "pharmacist":
            return "处方审核仅限药师操作。医生请使用药品查询功能，患者请咨询医师。"
        return await _review_prescription(patient_name)

    return await _query_drug_info(message, patient_name)
```

**角色系统（4 角色枚举）：**

**文件：** `src/medical_agent/governance/access_control.py:9-14`

```python
class UserRole(str, Enum):
    PATIENT = "patient"
    DOCTOR = "doctor"
    PHARMACIST = "pharmacist"
    ADMIN = "admin"
```

---

## 5. 药名提取（LLM）

**文件：** `src/medical_agent/orchestration/supervisor.py:503-523`

这是整个 Pipeline 中**唯一需要 LLM 做"理解"的环节**——从自然语言中提取药品名，处理缩略名、商品名、口语变体。

```python
async def _extract_drug_name(message: str) -> str:
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

    resp = llm.invoke(prompt)
    # 解析 JSON → 返回药品名
```

**覆盖验证：**

| 用户输入 | LLM 提取 | HISAdapter 别名匹配 |
|------|------|------|
| "二甲双胍说明书" | "二甲双胍" | 别名命中 → 盐酸二甲双胍片 |
| "格华止怎么吃" | "格华止" | 别名命中 → 盐酸二甲双胍片 |
| "那个降糖的药" | "二甲双胍" | LLM 推理 + 别名匹配 |
| "布洛芬和阿莫西林能一起吃吗" | "布洛芬" | 别名命中 |

---

## 6. 路径A：药品查询（医生/药师）

**文件：** `src/medical_agent/orchestration/supervisor.py:526-641`

### 6.1 流程

```
用户: "张三可以使用二甲双胍吗"（已通过 intent=drug + role=doctor）
  │
  ├─ ① LLM 提取药名（_extract_drug_name）
  │     → "二甲双胍"
  │
  ├─ ② HISAdapter.search_drugs("二甲双胍")
  │     → 别名匹配 → {name:"盐酸二甲双胍片", spec:"0.5g×30片", stock:400, price:22.0, ...}
  │
  ├─ ③ HISAdapter.get_patient_info("张三")（如有患者名）
  │     → 当前处方：[硝苯地平, 阿莫西林, 二甲双胍]
  │     → 过敏史：[青霉素, 磺胺类]
  │
  ├─ ④ GraphRAG.check_drug_interaction（如有多个药物）
  │     → 二甲双胍 + 硝苯地平（指南/明确）: 无显著相互作用
  │     → 二甲双胍 + 阿莫西林（文献/潜在）: 无显著相互作用
  │
  ├─ ⑤ LISAdapter 查检验指标（药物-检验联动）
  │     → Cr 82μmol/L (参考44-133) [正常]
  │     → 血糖 6.1mmol/L (参考3.9-6.1) [正常]
  │
  └─ ⑥ LLM 格式化输出（有患者时 → 临床评估格式，无患者时 → 信息服务格式）
```

### 6.2 有患者上下文的输出格式

```python
# supervisor.py:585-609
prompt = f"""你是临床药学助手。仅根据以下数据评估该患者是否可以使用此药。

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

院内信息：{drug_detail.get('spec','')} | {drug_detail.get('manufacturer','')} | 库存{drug_detail.get('stock','')}

免责声明：AI建议仅供参考，请以临床诊断和药师审核为准。"""
```

**设计原则：** 结论置顶（药品名 + ✓/⚠/✗）→ 患者信息 → 逐条评估依据 → 院内信息压底。医生 1 秒看到核心判断，需要细节时向下看。

### 6.3 无患者上下文的输出格式

```python
prompt = f"""仅根据以下数据回答药品查询，不要添加推断。
...
【院内信息】规格：...，厂商：...，库存：...盒，价格：¥...
【用药提醒】{{1-2句通用用药提醒}}
免责声明：..."""
```

---

## 7. 路径B：处方审核（仅药师）

**文件：** `src/medical_agent/orchestration/supervisor.py:396-490`

### 7.1 流程

```
药师: "审核张三的处方"
  │
  ├─ ① HISAdapter.get_patient_info("张三")
  │     → 处方: [硝苯地平 30mg qd, 二甲双胍 500mg bid]
  │     → 过敏史: [青霉素, 磺胺类]
  │
  ├─ ② 逐对 GraphRAG.check_drug_interaction()
  │     → 硝苯地平 + 二甲双胍（指南/明确）: 无显著相互作用
  │     → 所有对均已检查
  │
  ├─ ③ 过敏史交叉检查（程序逻辑，不依赖 LLM）
  │     → 二甲双胍 不含 青霉素/磺胺类 ✓
  │     → 硝苯地平 不含 青霉素/磺胺类 ✓
  │
  ├─ ④ LISAdapter 查检验指标（药物-检验联动）
  │     → Cr 82μmol/L [正常] — 二甲双胍无需调整剂量
  │
  └─ ⑤ LLM 1 次生成审核报告（仅做语言组织）
```

### 7.2 审核报告输出格式

```
处方审核报告

患者：张三
过敏史：青霉素、磺胺类

审核处方：
  P20250101 (内分泌科): 盐酸二甲双胍片 500mg bid, 硝苯地平控释片 30mg qd [待审核]

相互作用：未检测到显著相互作用

审核意见：处方合理，无过敏风险，无相互作用禁忌。可以调配。
（AI辅助审核，请药师最终确认）
```

### 7.3 核心代码

```python
# supervisor.py:396-490
async def _review_prescription(patient_name: str) -> str:
    # ① 获取患者处方 + 过敏史
    patient_info = await HISAdapter.get_patient_info(patient_name)
    prescriptions = patient_info["prescriptions"]

    # ② 提取所有药品名
    all_drugs = set()
    for p in target:
        for d in p.get("drugs", []):
            all_drugs.add(d["name"])

    # ③ 逐对检查相互作用（程序遍历，不依赖 LLM）
    drug_list = list(all_drugs)
    interactions = []
    for i in range(len(drug_list)):
        for j in range(i + 1, len(drug_list)):
            result = await rag.check_drug_interaction(drug_list[i], drug_list[j])
            if result and "无" not in str(result):
                interactions.append(f"  {drug_list[i]} + {drug_list[j]}: {result}")

    # ④ 过敏史检查（程序逻辑）
    allergies = patient_info.get("allergies", [])
    allergy_warning = ""
    for d in all_drugs:
        for allergy in allergies:
            if allergy in d:
                allergy_warning += f"\n  ⚠ {d}：患者有{allergy}过敏史！"

    # ⑤ LLM 1 次生成审核报告
    prompt = f"""你是药学审核助手。仅根据以下数据生成处方审核报告。
...
审核意见：{{1-2句话，是否通过审核}}
（AI辅助审核，请药师最终确认）"""

    resp = llm.invoke(prompt)
    return resp.content.strip()
```

---

## 8. 数据层

### 8.1 HISAdapter — 药品主数据 + 患者信息

**文件：** `src/medical_agent/adapters/his.py`

#### 药品数据（8 个，含别名）

```python
# L11-76
MOCK_DRUGS = [
    {"drug_code": "DRUG001", "name": "阿莫西林胶囊", "aliases": ["阿莫西林"], "spec": "0.5g×24粒", "stock": 500, "price": 12.50, "manufacturer": "华北制药"},
    {"drug_code": "DRUG004", "name": "盐酸二甲双胍片", "aliases": ["二甲双胍", "格华止"], "spec": "0.5g×30片", "stock": 400, "price": 22.00, "manufacturer": "中美施贵宝"},
    # ... 共 8 个药品，每个含 1-3 个别名
]
```

#### 患者处方 + 过敏史

```python
# L86-130
MOCK_PRESCRIPTIONS = [
    {"prescription_id": "P20250101", "patient_name": "张三", "patient_id": "P202501001",
     "drugs": [{"drug_code": "DRUG004", "name": "盐酸二甲双胍片", "dose": "500mg bid"},
               {"drug_code": "DRUG005", "name": "硝苯地平控释片", "dose": "30mg qd"}],
     "department": "内分泌科", "status": "待审核"},
    # ... 3 个处方
]

PATIENT_INFO = {
    "P202501001": {"patient_name": "张三", "allergies": ["青霉素", "磺胺类"],
                   "conditions": ["2型糖尿病", "高血压1级"]},
    "P20250210001": {"patient_name": "李四", "allergies": [],
                     "conditions": ["慢性胃炎"]},
}
```

#### 查询方法

| 方法 | 行 | 用途 |
|------|:--:|------|
| `search_drugs(keyword)` | L147-157 | 按名称/代码/别名模糊搜索 |
| `get_drug_by_code(code)` | L159-165 | 精确查询 |
| `get_patient_info(patient_name)` | L173-186 | 聚合处方+过敏史+病史 |

### 8.2 GraphRAGEngine — 药物相互作用（Neo4j 知识图谱）

**文件：** `src/medical_agent/engines/graph/graph_rag.py:382-409`

底层通过 Cypher 查询 Neo4j 知识图谱，检查两种药物是否共同关联同一疾病：

```python
# graph_rag.py:392-399 — check_drug_interaction 内部 Cypher
cypher = """
MATCH (d1:Drug {name: $drug_a})<-[:COMMON_DRUG|RECOMMEND_DRUG]-
      (disease:Disease)-[:COMMON_DRUG|RECOMMEND_DRUG]->(d2:Drug {name: $drug_b})
RETURN disease.name AS shared_disease, d1.name, d2.name
"""
```

**调用方式：** Pipeline 中逐对两两检查，结果解析证据等级：

```python
# supervisor.py:594-596
result = await rag.check_drug_interaction(drug_detail["name"], med)
if result and "无" not in str(result):
    evidence = "指南/明确" if any(k in str(result) for k in ("明确","显著","禁忌","CYP")) else "文献/潜在"
    patient_checks += f"  相互作用: {drug_a} + {drug_b}（{evidence}）: {result}"
```

**设计理由：** 通过共享疾病推断联合用药风险，而非硬编码相互作用列表。证据等级标注帮助医生区分"明确禁忌"和"理论风险"。

### 8.3 患者名提取（共享函数）

**文件：** `src/medical_agent/orchestration/supervisor.py:387-391`

```python
def _extract_patient_name(message: str) -> str:
    for p in _KNOWN_PATIENTS:
        if p in message:
            return p
    return ""
```

---

## 9. 完整代码调用链

### 医生查药："张三可以使用二甲双胍吗"

```
  ├─[入口] gradio_app.py:125  chat_send → POST /api/v1/chat/stream
  ├─[API]  chat.py:34         _stream_generator → orchestrate()
  │
  ├─[意图] intent_router.py:164  "可以吃吗" ∈ DRUG关键词 → intent=DRUG (0ms)
  │
  ├─[权限] supervisor.py:649    role=doctor → 通过 → _call_worker(role=doctor)
  ├─[路由] supervisor.py:58    intent=drug → _handle_drug(message, role=doctor)
  │
  ├─[路由] supervisor.py:378   "审核" not in msg → _query_drug_info(message, patient_name)
  │
  ├─[提取] supervisor.py:503   _extract_drug_name → LLM → "二甲双胍" (~3s)
  ├─[数据] his.py:147          search_drugs("二甲双胍") → 别名命中 (~0ms)
  ├─[数据] his.py:173          get_patient_info("张三") → 处方+过敏史 (~0ms)
  ├─[互作] supervisor.py:576   check_drug_interaction → GraphRAG+Neo4j (证据等级标注) (~0ms)
  ├─[检验] supervisor.py:585   get_patient_lis_report → Cr/血糖/CRP (~0ms)
  ├─[LLM]  supervisor.py:597   prompt → 临床评估 → qwen3.7-max (~5s)
  │
  └─[返回] chat.py:47          逐字 SSE event: token → Gradio Chatbot 渲染
```

**总 LLM 调用：2 次（药名提取 + 临床评估），总耗时 ~10s。**

### 药师审方："审核张三的处方"

```
  ├─[意图] intent_router.py:164  "处方" ∈ DRUG关键词 → intent=DRUG (0ms)
  │
  ├─[权限] supervisor.py:649    role=pharmacist → 通过
  ├─[路由] supervisor.py:58    intent=drug → _handle_drug(message, role=pharmacist)
  │
  ├─[路由] supervisor.py:379   "审核" in msg + role=pharmacist → _review_prescription("张三")
  │
  ├─[数据] his.py:173          get_patient_info("张三") → 处方+过敏史 (~0ms)
  ├─[互作] supervisor.py:430   for i,j pairs → check_drug_interaction() → Neo4j (证据等级) (~0ms)
  ├─[过敏] supervisor.py:436   for drug in drugs: for allergy in allergies (~0ms)
  ├─[检验] supervisor.py:451   get_patient_lis_report → Cr/血糖/CRP (~0ms)
  ├─[LLM]  supervisor.py:465   prompt → 审核报告 → qwen3.7-max (~5s)
  │
  └─[返回] chat.py:47          逐字 SSE event: token → Gradio Chatbot 渲染
```

**总 LLM 调用：1 次（审核报告），总耗时 ~5s。**

---

## 10. 文件索引

### 核心链路文件

| 文件 | 路径 | 行数 | 职责 |
|------|------|:--:|------|
| 编排+药物 | `src/medical_agent/orchestration/supervisor.py` | 681 | 角色权限、药名提取、药品查询、处方审核 |
| 意图路由 | `src/medical_agent/orchestration/intent_router.py` | 206 | DRUG 关键词 + LLM 增强分类 |
| 药品数据 | `src/medical_agent/adapters/his.py` | 209 | 8 药品+别名、患者处方+过敏史 |
| 角色定义 | `src/medical_agent/governance/access_control.py` | 147 | 4 角色枚举 |
| UI 入口 | `src/medical_agent/ui/gradio_app.py` | 375 | 角色下拉框、快捷操作按钮 |
| API 端点 | `src/medical_agent/api/routers/chat.py` | 62 | FastAPI SSE 端点 |

### 关键函数索引

| 函数 | 文件:行 | 用途 |
|------|------|------|
| `orchestrate()` | `supervisor.py:635` | 编排入口，角色拦截 |
| `_call_worker()` | `supervisor.py:48` | intent→pipeline 调度 |
| `_handle_drug()` | `supervisor.py:374` | 药物路由：审核 vs 查询 |
| `_extract_drug_name()` | `supervisor.py:503` | LLM 提取药名 |
| `_extract_patient_name()` | `supervisor.py:387` | 患者名提取（共享） |
| `_query_drug_info()` | `supervisor.py:526` | 药品查询 pipeline |
| `_review_prescription()` | `supervisor.py:396` | 药师处方审核 pipeline |
| `search_drugs()` | `his.py:147` | 药品名/别名/代码搜索 |
| `get_patient_info()` | `his.py:173` | 患者处方+过敏史查询 |
| `classify()` | `intent_router.py:147` | 4 步分类流水线 |

---

## 11. 与 ChatGPT 的本质区别

| 维度 | ChatGPT | 灵枢 Drug Agent |
|------|------|------|
| **药品数据** | 通用训练语料，可能过时 | 本院 HIS 实时库存/价格/规格 |
| **患者上下文** | 无患者数据 | 当前处方 + 过敏史 + 检验指标 |
| **药物相互作用** | 可能幻觉 | GraphRAG 知识库确定性查询 |
| **角色权限** | 任何人都能查 | patient→拒绝，doctor→仅查询，pharmacist→全权限 |
| **输出格式** | 自由文本 | 结论置顶+逐条评估+院内信息 |
| **临床评估** | 泛化建议 | "可以使用 ✓" + 过敏史/相互作用/肾功能逐项检查 |
| **法律合规** | 不清楚 | 患者侧直接拒绝，免临床风险；药师侧标注"请最终确认" |

**一句话总结：ChatGPT 告诉你"二甲双胍是什么"，Drug Agent 告诉你"张三现在能不能吃，需要做什么调整"。**
