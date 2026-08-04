
# 灵枢医疗 — 患者问诊 Agent 优化全记录

> 面试用：每条按「问题→方案→效果+代码位置」三段式。5 类共 17 项优化。

---

## 0. 优化一页速览

| # | 分类 | 优化项 | 问题 | 方案 | 效果 |
|:--:|------|--------|------|------|------|
| ① | 架构 | ReAct → 确定性编排 | 每次请求 3-5 次 LLM tool-calling 循环 | orchestrate() 直接意图分类 + Worker 调用 | 单轮 LLM 调用 5→2 |
| ② | 架构 | 无状态 → checkpoint | 每轮新建 Agent，症状归零死循环 | MemorySaver + thread_id 跨轮持久化 | 症状累积生效 |
| ③ | 架构 | Agent 缓存单例 | 每请求重新编译 StateGraph | _AGENT_CACHE 全局缓存 | 消除编译开销 |
| ④ | 收敛 | 计数 → 3因子加权 | len(symptoms)>=2 粗糙 | 维度40%+置信度30%+轮次30% | 精细化 |
| ⑤ | 收敛 | 概率→中文标签 | 患者看不懂85% | _prob_label() 映射可能性极高 | 患者可读 |
| ⑥ | 收敛 | 否定追踪自适应 | 用户说没有→连问5轮同类 | negation/detail 计数+3分支 | 死循环消失 |
| ⑦ | 收敛 | 延迟候选查询 | 1个症状查候选无意义 | len>=2 才调 _query_candidates | -20s/轮 |
| ⑧ | 收敛 | 科室动态警告 | 骨科输出腹痛警告 | _DEPT_WARNINGS 9科室 | 诊断匹配 |
| ⑨ | 性能 | 关键词预筛 | 每次调LLM(14s)分类意图 | _INTENT_KEYWORD_MAP 50+词 | -14s/轮 |
| ⑩ | 性能 | 删除重复LLM | _quick_check 有两段 try/except | 合并为一段 | -5s/轮 |
| ⑪ | 输出 | str()→结构化 | str(candidates)丢失字段 | 逐字段拼接 | 信息密度高 |
| ⑫ | 输出 | ChatGPT→Agent风格 | 温暖→800字作文 | 严格格式+禁止寒暄 | 200字报告 |
| ⑬-⑰ | Bug | 5个关键bug | JSON解析/dict提取/症状丢弃/污染/死循环 | 逐个修复 | 稳定运行 |

### 核心数据

`
               优化前        优化后
LLM调用/首轮     5次          2次        (-60%)
首轮耗时        ~45s         ~10s       (-78%)
收敛方式        5轮强制       2-3轮主动
症状累积        不生效         跨轮累积
输出风格        ChatGPT        Agent结构化
`

---

## 1. 架构优化

### ① ReAct 循环 → 确定性编排
**supervisor.py:69-97**

问题：主管编排层用 LangGraph ReAct——LLM 自主决定调用哪个 Worker，反复 tool-calling 3-5 轮。

方案：废除 ReAct。orchestrate() 直接：意图分类→急诊检测→权限检查→Worker调用。全流程只有 1 次 LLM（意图分类）。

效果：编排层 LLM 调用 3-5 次 → 1 次。确定性：可测试、可观测、零幻觉。

### ② 无状态 → 多轮 checkpoint
**agent.py:398-409,461-527**

问题：run_inquiry() 每次新 Agent + 新状态。用户消息是独立的上下文，症状从不累积——维度永远=1，必须 5 轮强制收尾。

方案：引入 LangGraph MemorySaver checkpoint。同一 session_id → thread_id。首轮传入完整初始状态，后续轮仅追加 message——其余字段（normalized_symptoms 等）从 checkpoint 自动恢复。

效果：症状跨轮累积。开发用 MemorySaver（内存），生产换 AsyncRedisSaver（Redis）。

### ③ Agent 缓存单例
**supervisor.py:14,22-45**

问题：每次请求重建 Agent 实例。方案：_AGENT_CACHE = {} 模块级字典，懒加载+单例。效果：消除编译开销。

---

## 2. 收敛策略优化

### ④ 简单计数 → 3 因子加权评分
**agent.py:337-363**

问题：len(symptoms)>=2 太粗糙——1个症状永远不收敛（5轮强制），3个症状立刻收尾（信息可能不足）。

方案：
`
score = 维度满足(>=3) × 0.4 + 置信度满足(top1>=0.5,gap>=0.15) × 0.3 + 轮次(>=2) × 0.3
>=0.6 收敛，>=5轮强制停止
`

设计理由：
- 维度 40%最高：症状数量客观，LLM概率不准不校准
- 置信度 30%：gap>=0.15 防止多个疾病概率接近的虚假满足
- 轮次 30%：至少追问1轮才考虑收尾

### ⑤ 概率 → 中文标签
**agent.py:214-224**

85% → 可能性极高。按 0.8/0.6/0.4/0.2 阈值分 5 档。

### ⑥ 否定追踪 + 自适应追问（方案D）
**agent.py:103-114,379-420**

问题：单一追问策略——永远问有没有伴随症状。用户持续回答没有 → 连问 5 轮同类 → 死循环。

方案：新增 2 个状态字段：
- negation_count：连续否定次数（_extract_symptoms 维护）
- symptom_detail_count：已收集细节数（有效回答时+1，否定时保留）

_quick_check 中 3 分支决策：
- ① 默认：追问伴随症状
- ② negation>=2 且 detail<2：切换问症状细节（时长/程度/诱因）
- ③ detail>=2 且 negation>=2 且 rounds>=3：提前收敛

状态机示例：
`
膝关节痛 → ①追伴随 → 没有(neg=1) → ①追伴随 → 没有(neg=2)
→ ②追细节 → 几天(detail=1) → ②追细节 → 加重(detail=2) → ③收敛
`

### ⑦ 延迟候选疾病查询
**agent.py:369-372**

症状<2时不调 _query_candidates()（~20s）。首轮省 20s。

### ⑧ 按科室动态紧急提醒
**agent.py:268-280**

_DEPT_WARNINGS 9 科室映射表。骨科 → 关节红肿，心血管 → 胸痛，消化 → 黑便。

---

## 3. 性能优化

### ⑨ 意图分类关键词预筛
**intent_router.py:50-84,164-167**

问题：每次调 LLM(14s)分类意图。90% 消息是肚子痛、有恶心——关键词即可判断。

方案：_INTENT_KEYWORD_MAP（5种意图×10+词）。遍历顺序 drug→report→knowledge→operation→inquiry（越具体越优先）。命中直接返回 confidence=0.85，全未命中才回退 LLM。

效果：90%+ 请求省 14s。首轮耗时 45s → 10s (-78%)。

### ⑩ 删除重复 LLM 调用
**agent.py:396-405**

_quick_check 追问生成中两段重复 try/except（merge artifact），删一段省 5s/轮。

---

## 4. 输出质量优化

### ⑪ str(candidates) → 结构化拼接
**agent.py:252-260**

旧：candidates_text = str(candidates) → [{'name':...,'probability':...}] 一坨
新：逐字段 f{name}（{label}，{dept}）：{reasoning}

### ⑫ ChatGPT → Agent 风格
**agent.py:286-302,313-326**

旧 prompt：要求专业、温暖、易懂 → 输出 您好！了解到...别太担心... （800字作文）
新 prompt：不要添加推断、不要寒暄、固定模板（不超过300字）
LLM 失败时程序直接拼装（无 LLM 依赖）

---

## 5. Bug 修复（5个）

| # | Bug | 根因 | 修复 | 位置 |
|:--:|------|------|------|------|
| ⑬ | JSON解析失败 | {[^{}]*} 遇嵌套截断 | {[\s\S]*} 贪婪匹配 | agent.py:158 |
| ⑭ | dict→string | Gradio 6.x dict被str() | isinstance(dict)分支取content | agent.py:93-95 |
| ⑮ | 症状丢弃 | still_unmatched未入all_standard | + still_unmatched | symptom_normalizer.py:211 |
| ⑯ | 追问污染 | 收敛时旧handoff未清除 | = None | agent.py:375-378 |
| ⑰ | 配额死循环 | API 403→symptoms=[]→循环 | 兜底+程序拼装 | 两处 |

---

## 6. 面试话术（3分钟）

我做的核心是一个患者问诊 Agent，基于 LangGraph + qwen3.7-max。

架构上三个关键决策：
① 不用 ReAct，改成确定性编排——意图分类+直接Worker调用，LLM调用5次→2次
② 引入 MemorySaver checkpoint 多轮有状态——症状跨轮累积，解决死循环
③ 意图分类加关键词预筛——90%消息直接命中，省14s/轮

收敛策略：三层加权评分（维度40%+置信度30%+轮次30%），阈值0.6。维度权重最大因为症状数量客观，LLM概率不校准。

单症状场景：否定追踪+自适应追问。2次否定后自动切换问细节，2个细节后提前收敛，永不陷入死循环。

输出：LLM从温暖→严格格式禁止寒暄，800字作文→200字结构化报告。LLM失败有纯程序兜底。

最终效果：单轮 45s→10s，LLM调用减半，2-3轮主动收尾，123测试全绿。

### 一句话技术亮点

> 确定性编排 + checkpoint多轮状态 + 关键词预筛 + 3因子加权收敛 + 否定追踪自适应追问 —— 五层优化把LLM调用压到极限，同时保证输出质量和可用性。
