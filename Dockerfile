# -*- coding: utf-8 -*-
# 灵枢医疗多智能体 - Docker 镜像构建文件
FROM python:3.12-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 拷贝 requirements.txt 并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目源代码
COPY src/ /app/

# 设置工作目录
WORKDIR /app

# 暴露服务端口
EXPOSE 8080

# 启动 FastAPI 服务
CMD ["uvicorn", "medical_agent.main:app", "--host", "0.0.0.0", "--port", "8080"]
