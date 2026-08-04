# -*- coding: utf-8 -*-
# Neo4j异步驱动 - 医学知识图谱存储
from neo4j import AsyncDriver, AsyncGraphDatabase

from medical_agent.core.config import get_settings

_driver: AsyncDriver | None = None


def get_neo4j_driver() -> AsyncDriver:
    """获取Neo4j异步驱动单例（延迟初始化）"""
    global _driver
    if _driver is None:
        settings = get_settings()
        _driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
    return _driver


async def check_neo4j_health() -> bool:
    """检查Neo4j服务健康状态"""
    try:
        driver = get_neo4j_driver()
        await driver.verify_connectivity()
        return True
    except Exception:
        return False


async def close_neo4j_driver() -> None:
    """关闭Neo4j驱动连接"""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
