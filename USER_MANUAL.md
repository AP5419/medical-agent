# 灵枢医疗多智能体系统 — 使用说明书

**版本**：1.0.0 | **Python**：≥ 3.11

---

## 一、系统概述

灵枢医疗多智能体系统是一套基于 8 层架构的医疗 AI 助手，采用 FastAPI + LangGraph 构建，
集成 MySQL、Redis、Milvus、Neo4j、MinIO 五大中间件，面向**患者、医生、管理员**三类用户
提供**智能问诊、报告解读、药物咨询、知识问答、运营分析**等 5 类服务。

### 8 层架构

```
用户接入层 (Gradio UI)
    ↓
接口层 (FastAPI REST + SSE 流式)
    ↓
编排调度层 (Supervisor Agent + 意图路由)
    ↓
智能体层 (问诊/报告/药品/知识/运营 5 个 Agent)
    ↓
能力引擎层 (RAG / GraphRAG / MS GraphRAG / VLM / NL2SQL / Memory / MinerU)
    ↓
数据源层 (ORM 模型 + HIS/LIS/EMR 模拟适配器)
    ↓
模型层 (LLM Provider + Embedding Provider)
    ↓
数据治理层 (审计追踪 / 数据脱敏 / RBAC 权限)
──────────────────────────────────────────
基础设施层 (MySQL 8.0 / Redis / Milvus / Neo4j / MinIO)
```

---

## 二、环境要求

| 组件 | 版本 | 用途 |
|------|------|------|
| Python | 3.11+ | 运行环境 |
| MySQL | 8.0 | 结构化业务数据（患者、药品、问诊记录） |
| Redis Stack | 7.4 | 会话缓存 + LangGraph 短期记忆 |
| Milvus | 2.6 | 向量检索（RAG） + 患者长期记忆 |
| Neo4j | 5.20 | 医学知识图谱（疾病-症状-药品-科室） |
| MinIO | latest | 文件/报告上传存储 |
| DeepSeek API | - | 对话大模型 |
| DashScope API | - | 文本嵌入 + 视觉语言模型 |
| MinerU | 3.0+ | PDF/DOCX/PPTX/图片 文档解析 |

---

## 三、快速开始

### 3.1 启动基础设施（Docker）

```bash
cd medical-agent
docker compose up -d
```

验证各服务状态：

| 服务 | 访问地址 | 账号/密码 |
|------|---------|-----------|
| MySQL | localhost:15308 | medical / medical123 |
| RedisInsight | http://localhost:8001 | 无 |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| Attu (Milvus) | http://localhost:13000 | 无 |
| Neo4j Browser | http://localhost:7474 | neo4j / medical123 |

### 3.2 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入 API Key：

```ini
DASHSCOPE_API_KEY=sk-xxxxxxxx     # 阿里云 DashScope（对话+嵌入+视觉）
```

### 3.3 安装依赖

```bash
pip install -e .
```

### 3.4 初始化数据

按以下顺序依次执行：

```bash
# 1. MySQL 结构化数据（科室/症状/药品/疾病）
python scripts/init_mysql.py

# 2. Neo4j 医学知识图谱
python scripts/init_neo4j.py

# 3. Milvus 症状向量索引
python scripts/init_milvus.py

# 4. MS GraphRAG 文档索引（可选，需要大模型 API Key）
#    先设置环境变量：
#    $env:GRAPHRAG_API_KEY="sk-xxx"           (Windows PowerShell)
#    $env:GRAPHRAG_API_BASE="https://api.deepseek.com"
python scripts/init_graphrag.py
```

### 3.5 启动应用

```bash
# FastAPI 后端 → http://localhost:8080
uvicorn medical_agent.main:app --port 8080 --reload

# Gradio Web UI → http://localhost:7860
python -m medical_agent.ui.gradio_app
```

验证：`curl http://localhost:8080/health` → `{"status": "ok"}`

---

## 四、API 接口参考

### 4.1 认证接口 `/api/v1/auth`

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| POST | `/register` | 无 | 用户注册 |
| POST | `/login` | 无 | 用户登录 |
| POST | `/refresh` | 无 | 刷新 Token |
| GET | `/me` | Bearer Token | 获取当前用户信息 |

**注册请求**：
```json
{
    "username": "patient01",
    "password": "123456",
    "role": "patient",
    "real_name": "张三"
}
```
- `role` 可选值：`patient`（患者）、`doctor`（医生）、`admin`（管理员）

**登录响应**：
```json
{
    "access_token": "eyJhbGciOi...",
    "refresh_token": "eyJhbGciOi...",
    "user": {
        "id": 1,
        "username": "patient01",
        "role": "patient",
        "real_name": "张三"
    }
}
```
- Access Token 有效期：24 小时
- Refresh Token 有效期：7 天

**示例**：
```bash
# 注册
curl -X POST http://localhost:8080/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"patient01","password":"123456","role":"patient"}'

# 登录
curl -X POST http://localhost:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"patient01","password":"123456"}'
```

### 4.2 对话接口 `/api/v1/chat`

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| POST | `/` | Bearer Token | 普通对话 |
| POST | `/stream` | Bearer Token | SSE 流式对话 |

**请求体**：
```json
{
    "user_id": "1",
    "session_id": "abc123",
    "message": "我头疼发烧怎么办？",
    "patient_id": null
}
```

**非流式响应**：
```json
{
    "content": "根据您的描述，您可能...",
    "thread_id": "1:abc123"
}
```

**流式响应**（SSE 格式）：
```
event: token
data: 根据

event: token
data: 您的

...

event: done
data:
```

### 4.3 文件上传 `/api/v1/upload`

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| POST | `/report` | Bearer Token | 上传报告图片/PDF/DICOM |
| GET | `/report/{name}` | Bearer Token | 下载报告文件 |
| POST | `/document` | Bearer Token | 上传 PDF/DOCX/PPTX/XLSX 文档（MinerU 解析为 Markdown） |

- `/report` 支持格式：JPEG、PNG、PDF、DICOM，上限 20 MB
- `/document` 支持格式：PDF、DOCX、PPTX、XLSX，上限 50 MB
- `/document` 解析流程：上传 → MinIO 存储 → MinerU 解析 → 返回 Markdown 文本

### 4.4 健康检查

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| GET | `/health` | 无 | 简单存活检测 |
| GET | `/health/deps` | 无 | 五大中间件健康检查 |

```json
// /health/deps 响应示例
{
    "status": "ok",
    "dependencies": {
        "mysql": {"ok": true, "error": ""},
        "redis": {"ok": true, "error": ""},
        "minio": {"ok": true, "error": ""},
        "milvus": {"ok": true, "error": ""},
        "neo4j": {"ok": true, "error": ""}
    }
}
```

---

## 五、5 个智能体使用说明

### 5.1 问诊 Agent（分诊导诊）

- **适用角色**：患者
- **触发条件**：用户描述身体不适、症状、不舒服
- **工作流程**：加载患者档案 → 急诊关键词检测 → **三层症状标准化（LLM口语→术语 → Neo4j精确匹配 → Milvus语义兜底）** → 候选疾病查询 → 追问鉴别症状 → 诊断结论
- **示例对话**：
  - "我头痛三天了，还发烧" → 系统追问 → 推荐科室和可能疾病
  - "胸口疼，喘不上气" → 急诊识别 → 提示拨打 120

### 5.2 报告解读 Agent

- **适用角色**：患者、医生
- **触发条件**：上传或讨论检验报告、影像结果
- **工具**：VLM 图片分析 + LIS 检验数据查询 + 指标知识库
- **示例**：
  - "帮我看看这个血常规报告"（上传图片）
  - "我的谷丙转氨酶偏高是什么意思？"

### 5.3 药物咨询 Agent

- **适用角色**：患者、医生、药师
- **触发条件**：询问用药、处方、药物相互作用
- **服务层级**：
  - **浅层**（患者/医生）：药品查询、注意事项、相互作用初判
  - **深层**（药师）：多跳图谱推理、处方安全性审查
- **示例**：
  - "阿莫西林和布洛芬能一起吃吗？"
  - "审一下这个处方：阿莫西林+头孢克肟+布洛芬"

### 5.4 知识问答 Agent

- **适用角色**：所有用户
- **触发条件**：询问疾病知识、医学术语、临床指南
- **5 个工具**：
  1. 医学文档检索（Milvus 向量搜索）
  2. 知识图谱查询（Neo4j 图数据库）
  3. 文档语料库搜索（MS GraphRAG 非结构化文档）
  4. 文档解析（MinerU PDF/DOCX → Markdown）
  5. 医学术语通俗解释
- **示例**：
  - "高血压的诊断标准是什么？"
  - "二甲双胍的常见副作用有哪些？"

### 5.5 运营数据 Agent

- **适用角色**：管理员
- **触发条件**：查询医院运营统计数据
- **安全限制**：仅允许 SELECT 聚合查询、自动脱敏、禁止访问患者信息
- **示例**：
  - "本月各科室门诊量排名"
  - "最近一周的药品使用量 TOP10"

---

## 六、核心引擎说明

### 6.1 医学 RAG 引擎

- **技术**：Milvus 向量检索 + HyDE 假设文档增强
- **流程**：查询改写（口语→医学术语）→ HyDE 生成 → Embedding → 向量搜索 → 重排序
- **向量维度**：1024（DashScope text-embedding-v3）

### 6.2 知识图谱引擎

- **技术**：Neo4j 图数据库 + LLM 生成 Cypher（NL2Cypher）
- **图谱规模**：7 类节点（疾病、症状、药品、科室、检查、食物、厂商）、8 种关系
- **关系类型**：HAS_SYMPTOM / BELONGS_TO / COMMON_DRUG / RECOMMEND_DRUG / NEED_CHECK / ACOMPANY_WITH / DO_EAT / NO_EAT

#### 6.2.1 NL2Cypher 查询正确性保障（5 层验证）

Cypher 由 LLM 生成，存在语法错误、幻觉标签、语义偏差等风险。系统采用纵深防御策略：

| 层 | 机制 | 成本 | 拦截场景 |
|:--:|------|------|------|
| ① | **Prompt 约束** | LLM ×1 | 注入完整 Schema + 标签/关系白名单列表到生成上下文 |
| ② | **格式检查** | 零 | 非 MATCH 开头 → 拒绝；代码块标记自动清理 |
| ③ | **标签/关系白名单** | 零（纯正则） | 拦截 `HAS_SYPMTOM` 等拼写错误，对照预定义的合法标签/关系列表 |
| ④ | **EXPLAIN 预编译** | Neo4j 计划层 | 语法校验（不实际执行查询） |
| ⑤ | **语义回译校验** | LLM ×2 (~400 token) | Cypher→NL 解释是否匹配用户意图。**当前为 soft-warning**（记录日志不拦截），积累数据后评估升级为硬拦截 |

**设计说明**：第⑤层采用 soft-warning 而非硬拦截，因为 LLM-as-Judge 自身的准确率未经评估，假阳性可能误拦正确查询。第③层白名单是零成本纯正则，弥补 EXPLAIN 不校验标签/关系是否真实存在的缺陷（Neo4j 是 schema-optional，不存在的标签不会报错）。

### 6.3 MS GraphRAG 引擎

- **技术**：Microsoft GraphRAG 官方包（Local Search + Global Search）
- **数据源**：25 份医疗文档（15 药品说明书 + 5 临床指南 + 5 医学科普）
- **索引配置**：`settings.yaml`，分块大小 800，重叠 100

### 6.4 NL2SQL 引擎

- **技术**：LLM 生成 SQL → 安全校验 → MySQL 执行 → 结果脱敏
- **安全层**：
  - 仅允许 SELECT 语句
  - 禁止 DROP / DELETE / INSERT / UPDATE / ALTER / CREATE / TRUNCATE
  - 自动脱敏：手机号 `138****5678`、身份证 `110***********1234`
  - 查询超时 10 秒

### 6.5 VLM 视觉引擎

- **技术**：DashScope Qwen-VL 多模态模型
- **能力**：检验报告 OCR、X 光/CT 影像分析、处方单识别

### 6.6 记忆引擎

- **短期记忆**：Redis Stack（LangGraph Checkpointer），会话级上下文
- **长期记忆**：Milvus 向量存储（1024 维），跨会话患者健康档案

### 6.7 MinerU 文档解析引擎

- **技术**：OpenDataLab/MinerU 开源包（`pip install mineru`）
- **输入格式**：PDF、DOCX、PPTX、XLSX、图片
- **输出格式**：Markdown / JSON（LLM 可直接消费的结构化文本）
- **核心能力**：OCR 109 种语言、公式→LaTeX、表格→HTML、页眉页脚去除、阅读顺序保留
- **调用方式**：本地 CLI 解析（默认） + 远程 API 解析（`MINERU_API_URL` 配置）
- **使用入口**：`POST /api/v1/upload/document` 上传文档 → 自动解析 → 返回 Markdown

---

## 七、数据目录说明

### 7.1 结构化数据

| 数据 | 路径 | 格式 | 说明 |
|------|------|------|------|
| 医学知识库 | Neo4j 图数据库 | 7类节点 + 8种关系 | 疾病/症状/药品/科室/检查/食物/厂商 |
| MySQL | Docker 服务 | 4 张表 | departments / patients / consultations / users |
| 知识库源文件 | `data/raw/medical.json` | JSONL | 初始化 Neo4j 的数据源 |

### 7.2 非结构化文档（MS GraphRAG 数据源）

| 类别 | 数量 | 路径 |
|------|------|------|
| 药品说明书 (txt) | 15 份 | `data/documents/drug_instructions/*.txt` |
| 临床指南 (txt) | 5 份 | `data/documents/guidelines/*.txt` |
| 医学科普 (txt) | 5 份 | `data/documents/education/*.txt` |
| PDF 文档 | 5 份 | `data/documents/pdf/*.pdf` |
| **合计** | **30 份** | |

药品说明书列表：阿莫西林胶囊、布洛芬缓释胶囊、头孢克肟分散片、二甲双胍片、苯磺酸氨氯地平片、阿托伐他汀钙片、奥美拉唑肠溶胶囊、氯雷他定片、阿司匹林肠溶片、胰岛素注射液、硝苯地平控释片、复方甘草片、蒙脱石散、盐酸氨溴索口服液、青霉素钠注射液

---

## 八、环境变量完整列表

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `APP_NAME` | medical-agent | 应用名称 |
| `APP_ENV` | dev | 环境（dev/prod） |
| `APP_DEBUG` | true | 调试模式 |
| `DB_HOST` | localhost | MySQL 主机 |
| `DB_PORT` | 15308 | MySQL 端口 |
| `DB_USER` | medical | 数据库用户 |
| `DB_PASSWORD` | medical123 | 数据库密码 |
| `DB_NAME` | medical_db | 数据库名 |
| `REDIS_HOST` | localhost | Redis 主机 |
| `REDIS_PORT` | 6379 | Redis 端口 |
| `REDIS_PASSWORD` | （空） | Redis 密码 |
| `REDIS_DB` | 0 | Redis 数据库编号 |
| `MINIO_ENDPOINT` | localhost:9000 | MinIO 地址 |
| `MINIO_ACCESS_KEY` | minioadmin | MinIO 访问密钥 |
| `MINIO_SECRET_KEY` | minioadmin | MinIO 秘密密钥 |
| `MINIO_BUCKET` | knowledge-docs | 存储桶名称 |
| `MINIO_SECURE` | false | 是否 HTTPS |
| `MILVUS_HOST` | localhost | Milvus 主机 |
| `MILVUS_PORT` | 19530 | Milvus gRPC 端口 |
| `NEO4J_URI` | bolt://localhost:7687 | Neo4j Bolt 地址 |
| `NEO4J_USER` | neo4j | Neo4j 用户名 |
| `NEO4J_PASSWORD` | medical123 | Neo4j 密码 |
| `DASHSCOPE_API_KEY` | （空） | 阿里云 DashScope API Key（对话+嵌入+视觉共用） |
| `CHAT_MODEL` | qwen-plus | 对话大模型名称 |
| `BASE_URL_CHAT` | （空） | 对话 API 地址 |
| `EMBEDDING_MODEL` | text-embedding-v3 | 嵌入模型名称 |
| `VL_MODEL` | qwen-vl | 视觉语言模型名称 |
| `LOG_LEVEL` | DEBUG | 日志级别 |
| `LOG_DIR` | logs | 日志目录 |
| `MINERU_API_URL` | （空） | MinerU 文档解析 API |
| `MINERU_BACKEND` | （空） | MinerU 后端类型 |
| `MINERU_TIMEOUT` | 60 | MinerU 超时（秒） |

---

## 九、测试

```bash
# 运行全部 104 个测试
pytest tests/ -v

# 运行指定模块
pytest tests/test_core/test_security.py -v

# 带覆盖率报告
pip install pytest-cov
pytest tests/ --cov=src/medical_agent --cov-report=html
```

测试覆盖范围：

| 模块 | 测试数 | 内容 |
|------|--------|------|
| test_config | 7 | 配置加载、MySQL URL、端口 |
| test_security | 11 | bcrypt 哈希、JWT 签发/验证/过期 |
| test_exceptions | 10 | 异常状态码、继承关系 |
| test_desensitize | 29 | 手机号/身份证/姓名/文本脱敏 |
| test_access_control | 16 | RBAC 患者/医生/管理员权限 |
| test_adapters | 19 | HIS/LIS/EMR 模拟数据查询 |
| test_intent_router | 12 | 急诊关键词检测 |

---

## 十、项目目录结构

```
medical-agent/
├── docker-compose.yml          # Docker 基础设施编排
├── Dockerfile                  # 应用镜像构建文件
├── pyproject.toml              # Python 包配置 + 依赖
├── settings.yaml               # MS GraphRAG 索引配置
├── .env.example                # 环境变量模板
├── pytest.ini                  # 测试配置
├── alembic.ini                 # 数据库迁移配置
│
├── data/
│   ├── raw/medical.json        # 医学知识库数据
│   └── documents/              # 非结构化文档（30 份）
│       ├── drug_instructions/  #   药品说明书(txt) ×15
│       ├── guidelines/         #   临床指南(txt) ×5
│       ├── education/          #   医学科普(txt) ×5
│       └── pdf/                #   PDF 文档 ×5
│
├── scripts/                    # 数据初始化脚本
│   ├── init_mysql.py
│   ├── init_neo4j.py
│   ├── init_milvus.py
│   └── init_graphrag.py
│   └── generate_pdfs.py
│
├── alembic/                    # 数据库迁移
│   ├── env.py
│   └── versions/
│
├── tests/                      # 单元测试（104 个）
│   ├── conftest.py
│   └── test_core/
│       ├── test_config.py
│       ├── test_security.py
│       ├── test_exceptions.py
│       ├── test_desensitize.py
│       ├── test_access_control.py
│       ├── test_adapters.py
│       └── test_intent_router.py
│
└── src/medical_agent/          # 主源码包
    ├── main.py                 # FastAPI 应用入口
    ├── core/                   # 核心模块（配置/安全/异常/日志）
    ├── infra/                  # 基础设施（MySQL/Redis/Milvus/Neo4j/MinIO）
    ├── middleware/              # 中间件（认证/日志）
    ├── api/                    # Layer 2 接口层
    │   ├── deps.py
    │   └── routers/
    │       ├── auth.py
    │       ├── chat.py
    │       └── upload.py
    ├── orchestration/          # Layer 3 编排调度层
    │   ├── supervisor.py
    │   └── intent_router.py
    ├── agents/                 # Layer 4 智能体层（5 个 Agent）
    │   ├── inquiry/agent.py + symptom_normalizer.py
    │   ├── report/agent.py
    │   ├── drug/agent.py
    │   ├── knowledge/agent.py
    │   └── operation/agent.py
    ├── engines/                # Layer 5 能力引擎层（7 个引擎）
    │   ├── rag/medical_rag.py
    │   ├── rag/graph_rag_ms.py
    │   ├── rag/mineru_client.py
    │   ├── graph/graph_rag.py
    │   ├── vlm/vlm_client.py
    │   ├── nl2sql/nl2sql.py
    │   └── memory/memory.py
    ├── models/                 # Layer 6 数据源层 - ORM（4 张表）
    ├── adapters/               # Layer 6 数据源层 - 适配器
    ├── repositories/           # Layer 6 数据源层 - 仓储
    ├── providers/              # Layer 7 模型层
    ├── governance/             # Layer 8 数据治理层
    └── ui/                     # Layer 1 用户接入层（Gradio）
```

---

## 十一、常见问题

**Q：没有 Docker 能运行吗？**
A：部分可以。缺少 MySQL 无法存储数据、缺少 Neo4j 无法图谱查询。`

**Q：MS GraphRAG 不安装有什么影响？**
A：Knowledge Agent 的 `search_document_corpus` 工具返回"暂不可用"，其他功能不受影响。

**Q：如何新增用户？**
A：调用 `POST /api/v1/auth/register` 接口注册，或在 MySQL `users` 表直接插入（密码用 bcrypt 哈希）。

**Q：Token 过期怎么办？**
A：用 Refresh Token 调用 `POST /api/v1/auth/refresh` 获取新 Access Token。

**Q：如何查看日志？**
A：日志文件在 `logs/` 目录，按日期切割，保留 7 天。错误日志单独记录在 `error_*.log`。

---

## 十二、技术栈总表

### 12.1 应用框架

| 组件 | 版本 | 描述 |
|------|------|------|
| FastAPI | ≥0.135 | 异步 Web 框架，REST API + SSE 流式 |
| Starlette | ≥0.52 | FastAPI 底层框架，中间件基类 |
| Uvicorn | ≥0.42 | ASGI 服务器 |
| Pydantic | ≥2.12 | 数据校验与序列化 |
| Pydantic Settings | ≥2.13 | .env 驱动配置中心 |
| python-dotenv | ≥1.2 | 环境变量文件加载 |
| python-multipart | ≥0.0.22 | 文件上传解析 |

### 12.2 数据库与存储

| 组件 | 版本 | 描述 |
|------|------|------|
| MySQL | 8.0 (Docker) | 结构化业务数据（10 张表，utf8mb4） |
| SQLAlchemy | ≥2.0 | 异步 ORM |
| aiomysql | ≥0.2.0 | MySQL 异步驱动 |
| PyMySQL | ≥1.1 | MySQL 同步驱动（数据初始化脚本） |
| Alembic | ≥1.18 | 数据库版本迁移 |
| Redis Stack | 7.4.0-v3 (Docker) | 会话缓存 + LangGraph Checkpointer（短期记忆） |
| redis-py | ≥7.3 | Redis 异步客户端 |
| MinIO | RELEASE.2024-12-18T13-15-44Z (Docker) | 文件上传/下载，Milvus 存储后端 |
| minio-py | ≥7.2 | MinIO Python SDK |
| Milvus | v2.6.13 (Docker) | 向量数据库（RAG 语义检索 + 长期记忆） |
| PyMilvus | ≥2.6 | Milvus Python SDK |
| Neo4j | 5.20 (Docker) | 图数据库（医学知识图谱） |
| neo4j-python | ≥5.28 | Neo4j 异步 Bolt 驱动 |

### 12.3 LLM 与 Agent 框架

| 组件 | 版本 | 描述 |
|------|------|------|
| LangChain | ≥1.2 | LLM 应用框架 |
| LangChain Core | ≥1.2 | 核心抽象（BaseMessage、Tool 等） |
| LangChain Community | ≥0.4 | 社区集成（DashScopeEmbeddings 等） |
| langchain-openai | ≥1.1 | OpenAI 兼容协议适配（DashScope 通过此通道调用） |
| langchain-deepseek | ≥1.0 | DeepSeek API 适配（可选，切换模型时使用） |
| LangGraph | ≥1.1 | Agent 状态图编排（StateGraph） |
| langgraph-checkpoint | ≥4.0 | 检查点基类 |
| langgraph-checkpoint-redis | ≥0.4 | Redis 短期记忆 Checkpointer |
| DashScope | ≥1.25 | 阿里云灵积模型服务（对话 + 嵌入 + 视觉） |

### 12.4 文档处理

| 组件 | 版本 | 描述 |
|------|------|------|
| MinerU | ≥3.0 | PDF/DOCX/PPTX/XLSX/图片 → Markdown 结构化解析 |
| GraphRAG | ≥1.0 | Microsoft GraphRAG 官方包（非结构化文档知识图谱索引） |
| Pandas | ≥2.2 | 数据分析（GraphRAG 依赖） |
| PyYAML | ≥6.0 | YAML 配置解析（settings.yaml） |

### 12.5 前端与 UI

| 组件 | 版本 | 描述 |
|------|------|------|
| Gradio | ≥6.12 | Web 聊天界面（三角色登录/对话/快捷操作） |
| httpx | ≥0.28 | 异步 HTTP 客户端（Gradio 调用后端 API） |

### 12.6 安全

| 组件 | 版本 | 描述 |
|------|------|------|
| PyJWT | ≥2.12 | JWT Token 签发与校验（HS256，Access 24h / Refresh 7d） |
| bcrypt | ≥5.0 | 密码哈希与验证 |
| cryptography | ≥46.0 | 加密原语库（PyJWT 依赖） |

### 12.7 可观测性

| 组件 | 版本 | 描述 |
|------|------|------|
| Loguru | ≥0.7 | 结构化日志（按日切割、7 天保留、控制台彩色 + 文件双输出） |
| Sentry SDK | ≥2.54 | 异常监控上报（可选） |

### 12.8 测试

| 组件 | 版本 | 描述 |
|------|------|------|
| Pytest | ≥8.3 | 单元测试框架 |
| pytest-asyncio | ≥0.24 | 异步测试支持（asyncio_mode=auto） |

### 12.9 工具库

| 组件 | 版本 | 描述 |
|------|------|------|
| aiohttp | ≥3.10 | 异步 HTTP 客户端/服务端（MinerU API 模式 + Gradio 依赖） |
| tqdm | ≥4.67 | 进度条（数据初始化脚本） |
| greenlet | ≥3.3 | 协程支持（SQLAlchemy 异步依赖） |

---

## 十三、项目整体流程图

```
┌───────────────────────────────────────────────────────────────────────┐
│                       用户  患者 / 医生 / 管理员                         │
│                      Gradio UI  或  API 客户端                         │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│  L2  接口层  FastAPI (端口 8080)                                       │
│                                                                       │
│  POST /api/v1/chat  ·  AuthMiddleware(JWT)  ·  CORS  ·  LogMiddleware │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│  L3  编排调度层  Supervisor Agent                                      │
│                                                                       │
│  LLM Agent (system prompt 定义 5 类意图路由规则)                         │
│  ┌──────────────┐                                                     │
│  │ 7 个工具      │  call_inquiry_agent  call_report_agent               │
│  │              │  call_drug_agent      call_knowledge_agent            │
│  │              │  call_operation_agent search_memory  save_memory      │
│  └──────────────┘                                                     │
│                                                                       │
│  LLM 根据 system prompt 中的路由规则自主选择调用哪个工具                    │
│  调用前 → search_memory(查询患者历史)   调用后 → save_memory(保存关键信息)  │
│                                                                       │
│  SummarizationMiddleware: 4000token 或 6条消息触发摘要压缩               │
└──────┬───────────┬───────────┬───────────┬───────────────┬────────────┘
       │           │           │           │               │
       ▼           ▼           ▼           ▼               ▼
┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────────┐
│  L4 智能体层                                                       │
│                                                                   │
│ ╔══════════╗╔══════════╗╔══════════╗╔══════════╗╔══════════════╗ │
│ ║问诊Agent ║║报告Agent ║║药物Agent ║║知识Agent ║║ 运营Agent   ║ │
│ ║           ║║           ║║           ║║           ║║              ║ │
│ ║StateGraph║║VLM 分析  ║║HIS 药品   ║║Milvus RAG ║║NL2SQL 查询  ║ │
│ ║7 节点    ║║LIS 检验  ║║Neo4j 交互 ║║Neo4j 图谱 ║║Schema 展示  ║ │
│ ║          ║║PACS 影像 ║║多跳推理   ║║MS_GraphRAG║║              ║ │
│ ║          ║║指标知识  ║║处方审核   ║║MinerU 解析║║              ║ │
│ ║          ║║           ║║           ║║术语解释   ║║              ║ │
│ ╚══════════╝╚══════════╝╚══════════╝╚══════════╝╚══════════════╝ │
│                                                                   │
│  工具数: 0     工具数: 4   工具数: 4   工具数: 5   工具数: 2        │
└───┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬───────┘
    │      │      │      │      │      │      │      │      │
    ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼
┌───────────────────────────────────────────────────────────────────────┐
│  L5  能力引擎层  5 个引擎模块                                           │
│                                                                       │
│  ┌────────────────────┐  ┌────────────────────┐                       │
│  │ 医学 RAG (rag/)     │  │ 知识图谱 (graph/)   │                       │
│  │ ├ MedicalRAGEngine  │  │ └ GraphRAGEngine   │                       │
│  │ ├ MSGraphRAGEngine  │  │   NL2Cypher + 多跳  │                       │
│  │ └ MinerUClient      │  └────────────────────┘                       │
│  │  HyDE增强 / 社区检索  │                                            │
│  │  PDF解析→Markdown   │  ┌────────────────────┐                       │
│  └────────────────────┘  │ NL2SQL (nl2sql/)    │                       │
│                          │ └ NL2SQLEngine      │                       │
│  ┌────────────────────┐  │   SQL生成+安全校验   │                       │
│  │ 视觉 VLM (vlm/)     │  └────────────────────┘                       │
│  │ └ VLMClient         │                                              │
│  │   Qwen-VL 多模态    │  ┌────────────────────┐                       │
│  └────────────────────┘  │ 记忆 (memory/)       │                       │
│                          │ ├ Redis Checkpointer │                       │
│                          │ └ LongTermMemory     │                       │
│                          │   短期+长期双轨      │                       │
│                          └────────────────────┘                       │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│  L6  数据源层                                                          │
│                                                                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │  MySQL   │ │  Neo4j   │ │  Milvus  │ │  MinIO   │ │  Redis   │    │
│  │ 10 张表  │ │ 7节点8关系│ │ 向量集合 │ │ 文件存储 │ │ 缓存+会话 │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘    │
│                                                                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                 │
│  │HIS适配器 │ │LIS适配器 │ │EMR适配器 │ │PACS适配器│                 │
│  │药品+处方 │ │检验报告  │ │电子病历  │ │影像DICOM │                 │
│  │(模拟数据)│ │(模拟数据)│ │(模拟数据)│ │(模拟数据)│                 │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘                 │
│                                                                       │
│  ┌─────────────────────────────────────────────────────┐              │
│  │ 6 个 Repository (Department/Disease/Symptom/Drug/   │              │
│  │                  Patient/Consultation)               │              │
│  └─────────────────────────────────────────────────────┘              │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│  L7  模型层                                                            │
│                                                                       │
│  ┌─────────────────────┐    ┌──────────────────────────┐              │
│  │   LLM Provider       │    │  Embedding Provider      │              │
│  │   get_llm()          │    │  get_embedding_model()   │              │
│  │   qwen-plus (默认)   │    │  text-embedding-v3       │              │
│  │   自动检测 deepseek   │    │  1024 维向量             │              │
│  │   兼容 OpenAI 协议   │    │  DashScope API           │              │
│  └─────────────────────┘    └──────────────────────────┘              │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│  L8  数据治理层                                                        │
│                                                                       │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────────────────┐        │
│  │  审计追踪     │  │   数据脱敏      │  │   RBAC 权限控制     │        │
│  │  audit.py     │  │ desensitize.py │  │ access_control.py  │        │
│  │  每次查询     │  │  手机/身份证   │  │  3角色×多资源      │        │
│  │  记录日志     │  │  自动遮盖      │  │ 患者/医生/管理员   │        │
│  └──────────────┘  └────────────────┘  └─────────────────────┘        │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│  结果汇聚                                                              │
│                                                                       │
│  Supervisor → 保存记忆(Milvus) → JSON Response / SSE Stream → 用户     │
└───────────────────────────────────────────────────────────────────────┘
```

---

*使用说明书版本：v1.1 | 更新日期：2024-07*
