# -*- coding: utf-8 -*-
# Redis异步客户端 - 业务缓存与LangGraph检查点存储
import redis.asyncio as aioredis

from medical_agent.core.config import get_settings

_business_pool = None
_checkpointer_pool = None


def _get_business_pool():
    """业务缓存连接池（延迟初始化）"""
    global _business_pool
    if _business_pool is None:
        settings = get_settings()
        _business_pool = aioredis.ConnectionPool.from_url(
            f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
            password=settings.REDIS_PASSWORD or None,
            decode_responses=True,
            max_connections=50,
        )
    return _business_pool


def _get_checkpointer_pool():
    """LangGraph检查点连接池（延迟初始化，不自动解码）"""
    global _checkpointer_pool
    if _checkpointer_pool is None:
        settings = get_settings()
        _checkpointer_pool = aioredis.ConnectionPool.from_url(
            f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB + 1}",
            password=settings.REDIS_PASSWORD or None,
            decode_responses=False,
            max_connections=10,
        )
    return _checkpointer_pool


def get_redis_client() -> aioredis.Redis:
    """获取业务缓存Redis客户端（自动解码）"""
    return aioredis.Redis(connection_pool=_get_business_pool())


def get_checkpointer_redis() -> aioredis.Redis:
    """获取LangGraph检查点Redis客户端（不解码）"""
    return aioredis.Redis(connection_pool=_get_checkpointer_pool())


async def close_redis_pools() -> None:
    """关闭所有Redis连接池（应用关闭时调用）"""
    global _business_pool, _checkpointer_pool
    if _business_pool is not None:
        await _business_pool.disconnect()
        _business_pool = None
    if _checkpointer_pool is not None:
        await _checkpointer_pool.disconnect()
        _checkpointer_pool = None
