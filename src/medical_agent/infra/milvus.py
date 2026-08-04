# -*- coding: utf-8 -*-
# Milvus向量数据库连接 - 症状/文档/记忆向量存储

from pymilvus import Collection, connections

from medical_agent.core.config import get_settings

MILVUS_ALIAS = "medical_agent"

# 集合名称常量
COLLECTION_SYMPTOM = "medical_symptoms"
COLLECTION_DOCS = "medical_documents"
COLLECTION_MEMORY = "medical_memory"

_connected = False


def _ensure_connection() -> None:
    """确保 Milvus 连接已建立（connections.connect 是幂等的）"""
    global _connected
    if _connected:
        return
    settings = get_settings()
    connections.connect(
        alias=MILVUS_ALIAS,
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
    )
    _connected = True


def get_milvus_alias() -> str:
    """获取 Milvus 连接别名"""
    _ensure_connection()
    return MILVUS_ALIAS


def get_collection(name: str) -> Collection:
    """获取 Milvus 集合"""
    _ensure_connection()
    return Collection(name)


def check_milvus_health() -> bool:
    """检查 Milvus 服务健康状态"""
    try:
        _ensure_connection()
        return True
    except Exception:
        return False


def close_milvus_client() -> None:
    """断开 Milvus 连接"""
    global _connected
    try:
        connections.disconnect(MILVUS_ALIAS)
    except Exception:
        pass
    _connected = False
