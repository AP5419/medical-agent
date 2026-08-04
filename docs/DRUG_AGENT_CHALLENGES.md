# 药物咨询 Agent — 核心难点与优化过程

> 基于本项目实际开发中遇到的 7 个可验证工程难题。不涉及 mock 数据不完整等数据质量问题。

---

## 1. 概述

药物 Agent 与通用 LLM（ChatGPT）的 3 个核心差异决定了其工程复杂度：

| 差异 | 通用 LLM | 药物 Agent |
|------|------|------|
| **数据维度** | 训练语料中的通用药理 | HIS 实时库存 + 患者处方 + 过敏史 + LIS 检验 + GraphRAG 知识库 |
| **准确度要求** | 概率输出，可接受 10% 错误 | 药物建议不能出错——过敏/相互作用误判会致命 |
| **角色权限** | 无 | 患者被拒、医生仅查询、药师可审核——法律风险隔离 |

7 个难点均源于这三个核心差异。

---

## 2. 难点 1：LLM 调用静默失败

### 问题

持续调试多轮发现 `"二甲双胍可以吃吗"` 始终被判为 inquiry 而非 drug，修改 prompt 仍无效。3 个月未被发现的隐藏 bug。

### 根因

`INTENT_CLASSIFY_PROMPT` 中使用 `.format(message=message)` 填充模板，但模板中 JSON 示例用了单花括号 `{"intent": "意图类型"}`，Python `.format()` 将其解析为占位符 → `KeyError: '"intent"'` → `classify()` 中的 `except Exception: pass` 静默吞掉 → 返回默认 `IntentType.INQUIRY`。

**L3 LLM 意图分类从未真正执行过，所有未命中关键词的请求都靠默认值兜底。**

### 优化前

```python
# intent_router.py:98-100（旧）
请返回JSON格式：{"intent": "意图类型", "confidence": 0.0-1.0}

# classify():196-199
except Exception:
    pass          # ← 静默吞掉所有异常

return IntentResult(intent=IntentType.INQUIRY, confidence=0.3)
```

### 优化后

```python
# intent_router.py:100（修完）
请返回JSON格式：{{"intent": "意图类型", "confidence": 0.0-1.0}}
# ↑ {{ 和 }} 是 .format() 的字面花括号转义

# classify():197-199 — 补 debug 日志
except Exception as e:
    logger.warning(f"[意图] LLM分类失败: {e}")

logger.info(f"[意图] 分类结果: inquiry (default)")
```

### 效果

| | 修复前 | 修复后 |
|------|------|------|
| L3 LLM 分类 | 从未生效 | 正常执行 |
| "二甲双胍可以吃吗" | inquiry | drug |
| 异常可见性 | 静默吞掉 | 日志明确记录 |

---

## 3. 难点 2：ReAct 循环 → 确定性 Pipeline

### 问题

原 `drug/agent.py` 使用 LangChain `create_agent()` 创建 ReAct Agent，4 个工具（搜索药品/检查相互作用/多跳推理/处方审核）。LLM 自主决定调用哪个工具、如何组合、生成什么输出。每次查询 3-4 次 LLM 调用，响应 30s+。

### 根因

ReAct 循环中每步"思考→调工具→解读返回"都是一次 LLM API 调用。药品查询不需要"自主决策"——总是先查药名→查 HIS→查互作用→格式化输出。顺序固定，不需要 LLM 决定。

### 优化前

```python
# drug/agent.py:141-165
def create_drug_agent():
    llm = get_llm_qa()
    tools = [search_drug_info, check_drug_interaction,
             multi_hop_drug_reasoning, review_prescription_safety]
    return create_agent(llm, tools, system_prompt=DRUG_SYSTEM_PROMPT)
```

### 优化后

```python
# supervisor.py:526-641 — 药品查询确定性 Pipeline
async def _query_drug_info(message, patient_name):
    # ① LLM 提取药名（1次）— 仅此处需要 NLU
    drug_query = await _extract_drug_name(message)
    # ② HIS 查询（程序）
    drug_detail = await HISAdapter.search_drugs(drug_query)[0]
    # ③ 患者处方 + 过敏史（程序）
    patient_info = await HISAdapter.get_patient_info(patient_name)
    # ④ 药物相互作用（程序调 GraphRAG）
    result = await rag.check_drug_interaction(drug_a, drug_b)
    # ⑤ 检验指标（程序调 LISAdapter）
    lab_data = await LISAdapter.get_patient_lis_report(patient_name)
    # ⑥ LLM 格式化（1次）— 仅做语言组织
    resp = llm.invoke(prompt)
```

### 效果

| | ReAct | Pipeline |
|------|:--:|:--:|
| LLM 调用数 | 3-4 次 | 2 次（提取+格式化） |
| 响应时间 | 30-60s | ~8s |
| 工具调用 | LLM 自决 | 程序确定性 |
| 异常处理 | LLM 可能卡死 | 每步独立 try/except |

---

## 4. 难点 3：多源数据融合

### 问题

回答"张三是否可以吃二甲双胍"需要 4 个数据源的信息：
- HIS：药品库存/价格、患者当前处方
- HIS：患者过敏史（青霉素、磺胺类）
- GraphRAG+Neo4j：硝苯地平+二甲双胍的药物相互作用
- LIS：患者的 Cr/血糖/CRP 检验指标

这 4 个源的数据格式、字段名、查询方式完全不同，需要统一组装成一个 LLM prompt。

### 优化前

ReAct Agent 各自调工具，LLM 在对话历史中自行拼接——数据漏了也无感知。

### 优化后

```python
# supervisor.py:544-614 — 统一在 _query_drug_info 中依次调 4 个数据源
# ③ HIS 查处方 + 过敏史
patient_info = await HISAdapter.get_patient_info(patient_name)
all_meds = {d["name"] for p in prescriptions for d in p["drugs"]}
allergies = patient_info.get("allergies", [])

# ④ GraphRAG 逐对查相互作用
for med in all_meds:
    result = await rag.check_drug_interaction(drug_name, med)

# ⑤ LIS 按指标取最近值
TARGET = ("肌酐", "血糖", "超敏C反应蛋白")
latest_by_name = {}
for r in sorted(reports, key=lambda x: x["report_date"]):
    for ind in r["indicators"]:
        if ind["name"] in TARGET:
            latest_by_name[ind["name"]] = ind  # dict 覆盖 = 自动取最近

# ⑥ 拼入一个结构化 prompt
prompt = f"""
药品：{drug_name}
{patient_context}       # 处方 + 过敏史
{patient_checks}         # 相互作用 + 检验指标
"""
```

### 效果

| | 优化前 | 优化后 |
|------|------|------|
| 数据来源 | 仅通用药理 | HIS + GraphRAG + LIS 三源 |
| 缺失检测 | LLM 自由发挥 | 程序保证每步查询 |
| 异常降级 | 不可控 | 每步独立 try/except → 缺一个源仍输出 |

---

## 5. 难点 4：知识库返回值的语义解析

### 问题

GraphRAG 的 `check_drug_interaction()` 通过共享疾病推断相互作用，统一返回 `{"warning": "..."}`。但"两种药都用于高血压的常规联合用药"和"两种药都用于华法林方案的抗凝辅助有出血风险"——在知识库里是同一种返回结构，LLM 无法区分严重级别。

### 优化前

```python
# supervisor.py:595（旧）
result = await rag.check_drug_interaction(drug_a, drug_b)
if result and "无" not in str(result):
    patient_checks += f"  相互作用: {drug_a} + {drug_b}: {result}"
```

### 优化后

```python
# supervisor.py:597（新）
evidence = "指南/明确" if any(k in str(result) for k in ("明确","显著","禁忌","CYP")) else "文献/潜在"
patient_checks += f"  相互作用: {drug_a} + {drug_b}（{evidence}）: {result}"
```

### 效果

医生看到 `（指南/明确）` 的相互作用会认真对待，看到 `（文献/潜在）` 知道是理论风险。从"一律视为同等重要"变为"区分置信度"。

---

## 6. 难点 5：跨报告类型的指标聚合

### 问题

张三有 5 份检验报告（4 次血常规 + 1 次生化），`肌酐` 和 `血糖` 仅在 12/15 生化报告中，`超敏C反应蛋白` 在 12/22 血常规中。只取最新 1 份报告必然丢失部分指标。

### 根因

不同指标分属不同的报告类型——血常规/生化/尿常规的指标集互不相同。不能按"最新报告"来取，需要按"每个指标自己的最近测量值"来取。

### 优化前

```python
# supervisor.py:607（旧）
latest = sorted(reports, key=lambda r: r["report_date"])[-1]
for ind in latest["indicators"]:
    if ind["name"] in ("肌酐", "血糖", "超敏C反应蛋白"):
        # 12/22 血常规缺肌酐和血糖 → 这两个指标被丢弃
```

### 优化后

```python
# supervisor.py:609-611（新）
TARGET = ("肌酐", "血糖", "超敏C反应蛋白")
latest_by_name = {}
for r in sorted(reports, key=lambda x: x["report_date"]):  # 升序遍历
    for ind in r["indicators"]:
        if ind["name"] in TARGET:
            latest_by_name[ind["name"]] = ind  # dict 赋值 = 自动覆盖为较新值
for ind in latest_by_name.values():
    patient_checks += f"\n  检验: {ind['name']} {ind['value']}{ind['unit']}..."
```

### 效果

| 指标 | 旧（最新报告） | 新（按指标取最近值） |
|------|:--:|:--:|
| 肌酐 82 | ✗ 丢弃（在 12/15 生化中） | ✅ 保留（12/15 唯一一次测量） |
| 血糖 6.1 | ✗ 丢弃（在 12/15 生化中） | ✅ 保留（12/15 唯一一次测量） |
| CRP | ✅ 取 12/22 值（多次测量取最近） | ✅ 同上 |

### 算法保证

`sorted(reports, key=date)` 升序 + `dict[key] = value` 赋值覆盖 → 遍历完毕后每个 key 保留**最后被赋值的值**（即最近日期）。逻辑自洽，不需显式比较日期。

---

## 7. 难点 6：角色权限分层

### 问题

系统最初只有 patient/doctor/admin 三角色，无 pharmacist（药师）。且药物意图无角色检查——患者也能查药。

### 根因

药品咨询有法律风险：患者可能基于 AI 建议自行用药/停药。医生和药师场景不同——医生需决策辅助（"这个药能用吗"），药师需处方审核（"这张方子安全吗"）。

### 优化后

4 角色 × 3 操作的权限矩阵：

```python
# access_control.py:9-14
class UserRole(str, Enum):
    PATIENT = "patient"
    DOCTOR = "doctor"
    PHARMACIST = "pharmacist"    # 新增
    ADMIN = "admin"

# supervisor.py:646-651
if intent == "drug":
    if role == "patient":
        return "药物咨询仅限医生和药师使用，患者请咨询医师后用药。"
    return await _call_worker(intent, message, session_id, role=role)

# supervisor.py:374-382 — _handle_drug 内角色路由
_REVIEW_KEYS = ["审核", "审查", "处方", "核对", "检查处方", "审方"]
if any(k in message for k in _REVIEW_KEYS) and patient_name:
    if role != "pharmacist":
        return "处方审核仅限药师操作。"
    return await _review_prescription(patient_name)
```

### 效果

| 操作 | 患者 | 医生 | 药师 | admin |
|------|:--:|:--:|:--:|:--:|
| 药品查询 | ✗ | ✅ | ✅ | ✅ |
| 处方审核 | ✗ | ✗ | ✅ | ✗ |

---

## 8. 难点 7：临床输出格式工程

### 问题

相同的数据源，prompt 格式不同导致输出质量天差地别。

优化前（"用药提醒"格式）：
```
药品查询：盐酸二甲双胍片
【院内信息】规格：0.5g×30片，厂商：...
患者当前用药：硝苯地平、阿莫西林、盐酸二甲双胍。
【用药提醒】您有青霉素过敏史，请停药咨询医生...
```

主次不分——院内信息喧宾夺主，核心临床判断藏在正文末尾。

### 优化后（临床评估格式）

```python
# supervisor.py:585-609 — 有患者时 prompt 模板

{drug_detail['name']} — {{可以使用✓/慎用⚠/禁用✗}}

患者：{patient_name}
当前用药：{{列出}}
过敏史：{{列出}}

评估详情：
  · 过敏史：{{结论}}
  · 相互作用：{{结论}}
  · 肾功能（如有检验数据）：{{结论}}

院内信息：{spec} | {manufacturer} | 库存{stock}
免责声明：...
```

### 效果

| 维度 | 优化前 | 优化后 |
|------|------|------|
| 核心结论 | 藏在第 5 段 | **首行标题**（✓/⚠/✗） |
| 临床依据 | 笼统的"参见说明书" | 逐条列出过敏/互作用/肾功能 |
| 机构信息 | 最显眼位置 | 末尾一行压缩 |
| 阅读路径 | 从上到下扫完才能判断 | 1 秒看到结论，需要细节向下看 |

---

## 9. 总结

| 难点 | 类别 | 核心解法 | 关键文件 |
|------|:--:|------|------|
| LLM 调用静默失败 | 可靠性 | `{{` 转义 + 异常日志 | `intent_router.py:100` |
| ReAct→Pipeline | 架构 | 程序编排 → 2 次 LLM | `supervisor.py:526-641` |
| 多源数据融合 | 架构 | 4 源 → 1 prompt | `supervisor.py:544-614` |
| 知识库语义解析 | 可靠性 | 关键词 → 证据等级 | `supervisor.py:597` |
| 跨报告指标聚合 | 数据查询 | `sorted + dict merge` | `supervisor.py:609-611` |
| 角色权限分层 | 安全 | 4×3 权限矩阵 | `access_control.py` + `supervisor.py:646` |
| 临床输出格式 | 产品 | 结论置顶 + 逐条依据 | `supervisor.py:585-609` |

**一句话总结：药物 Agent 的难点不在 AI 能力本身，在于"把正确的数据以正确的格式送给 LLM，让 LLM 只做它擅长的事——语言组织"。**
