# -*- coding: utf-8 -*-
# 全局配置模块 - 从.env文件加载所有配置项
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """全局配置，自动读取.env文件"""

    # 应用配置
    APP_NAME: str = "medical-agent"
    APP_ENV: str = "dev"
    APP_DEBUG: bool = True

    # MySQL 配置
    DB_HOST: str = "localhost"
    DB_PORT: int = 15308
    DB_USER: str = "medical"
    DB_PASSWORD: str = "medical123"
    DB_NAME: str = "medical_db"

    # Redis 配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0

    # MinIO 配置
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "knowledge-docs"
    MINIO_SECURE: bool = False

    # Milvus 配置
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    # Neo4j 配置
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "medical123"

    # LLM 配置
    DASHSCOPE_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""
    CHAT_MODEL: str = "qwen3.7-max-2026-06-08"
    BASE_URL_CHAT: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    EMBEDDING_MODEL: str = "qwen3.7-text-embedding"
    VL_MODEL: str = "qwen-image-2.0-pro-2026-04-22"

    # 日志配置
    LOG_LEVEL: str = "DEBUG"
    LOG_DIR: str = "logs"

    # MinerU 配置
    MINERU_API_URL: str = ""
    MINERU_BACKEND: str = ""
    MINERU_TIMEOUT: int = 60

    @property
    def DATABASE_URL(self) -> str:
        """异步数据库连接字符串 (aiomysql)"""
        return (
            f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def DATABASE_SYNC_URL(self) -> str:
        """同步数据库连接字符串 (pymysql，用于Alembic)"""
        return (
            f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()
