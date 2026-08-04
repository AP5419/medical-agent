# -*- coding: utf-8 -*-
# Alembic 异步迁移环境配置 - 针对 MySQL 8.0
import asyncio
import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# 将项目 src 目录加入 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Alembic Config 对象，读取 alembic.ini
config = context.config

# 从 INI 文件获取日志配置
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 导入所有模型的 Base 元数据
from medical_agent.core.base_model import Base

# 导入所有 ORM 模型，确保 Base.metadata 包含全部表结构
from medical_agent.models.medical import (
    Department,
    Disease,
    Symptom,
    DiseaseSymptom,
    Drug,
    DrugDetail,
    DiseaseDrug,
    Patient,
    Consultation,
)
from medical_agent.models.user import User

# 从配置模块读取数据库连接字符串
from medical_agent.core.config import get_settings

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# target_metadata 指向所有模型元数据，供 autogenerate 使用
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """以离线模式运行迁移 —— 生成 SQL 脚本而非连接数据库执行。

    需要配置 URL，context.configure() 会生成一个仅输出 SQL 的上下文。
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """在给定的数据库连接上执行迁移。"""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """异步运行迁移 —— 创建异步引擎并通过连接执行。"""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """在线模式运行迁移 —— 连接到数据库并执行 DDL 语句。"""
    asyncio.run(run_async_migrations())


# 根据当前模式选择离线或在线迁移
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
