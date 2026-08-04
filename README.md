# 医疗多智能体系统

基于 **8 层架构** 的医疗 AI 多智能体系统，集成病历分析、智能问诊、药品知识库、报告解读与运营管理五大 Agent，为医疗机构提供全栈智能化解决方案。

## 技术栈

| 层次 | 技术选型 |
|------|----------|
| **API 层** | FastAPI + Pydantic |
| **中间件层** | 认证鉴权、日志追踪、请求限流 |
| **编排层** | LangGraph Supervisor 多智能体编排 |
| **智能体层** | 5 个 Agent：Inquiry（问诊）、Knowledge（知识）、Drug（药品）、Report（报告）、Operation（运营） |
| **引擎层** | RAG 检索、NL2SQL 自然语言转 SQL、VLM 多模态视觉、GraphRAG 知识图谱、Memory 记忆 |
| **治理层** | 访问控制、数据脱敏、审计日志 |
| **基础设施层** | MySQL 8.0、Redis Stack、Milvus、Neo4j、MinIO |
| **模型提供层** | DeepSeek / 通义千问 / 阿里云百炼 |

## 快速开始

### 1. 启动基础设施

```bash
docker compose up -d
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入 API Key 等信息
```

### 3. 安装项目依赖

```bash
pip install -e .
```

### 4. 启动开发服务器

```bash
uvicorn medical_agent.main:app --port 8080 --reload
```

服务启动后访问：
- API 文档: http://localhost:8080/docs
- Milvus 管理 (Attu): http://localhost:13000
- Neo4j 浏览器: http://localhost:7474
- MinIO 控制台: http://localhost:9001
- RedisInsight: http://localhost:8001
