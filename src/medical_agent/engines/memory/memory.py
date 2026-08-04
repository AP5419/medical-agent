# -*- coding: utf-8 -*-
"""记忆引擎 - LangGraph短期记忆检查点 + Milvus长期记忆管理"""

from typing import Optional

from medical_agent.providers.embedding import get_embedding_model


def create_checkpointer():
    """创建LangGraph的Redis异步检查点保存器，用于短期对话记忆"""
    from medical_agent.infra.redis import get_checkpointer_redis

    return get_checkpointer_redis()


class LongTermMemory:
    """长期记忆管理器，基于Milvus向量数据库存储和检索用户记忆"""

    # Milvus集合常量
    COLLECTION_MEMORY = "memory"
    EMBEDDING_DIM = 1024

    def __init__(self):
        """初始化长期记忆管理器，加载嵌入模型"""
        self.embedding_model = get_embedding_model()

    def _ensure_collection(self) -> bool:
        """确保Milvus记忆集合存在，不存在则创建

        Returns:
            集合是否可用
        """
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            connections,
            utility,
        )
        from medical_agent.core.config import get_settings

        settings = get_settings()

        try:
            connections.connect(
                alias="memory",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
            )

            if utility.has_collection(self.COLLECTION_MEMORY, using="memory"):
                return True

            # 定义Schema
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True,
                ),
                FieldSchema(
                    name="user_id",
                    dtype=DataType.VARCHAR,
                    max_length=100,
                ),
                FieldSchema(
                    name="content",
                    dtype=DataType.VARCHAR,
                    max_length=4096,
                ),
                FieldSchema(
                    name="memory_type",
                    dtype=DataType.VARCHAR,
                    max_length=50,
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.EMBEDDING_DIM,
                ),
            ]

            schema = CollectionSchema(fields, description="长期记忆集合")
            collection = Collection(
                name=self.COLLECTION_MEMORY,
                schema=schema,
                using="memory",
            )

            # 创建IVF_FLAT索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            }
            collection.create_index(
                field_name="embedding",
                index_params=index_params,
            )

            return True

        except Exception:
            return False

    def save_memory(
        self, user_id: str, content: str, memory_type: str = "general"
    ) -> bool:
        """保存用户长期记忆到Milvus

        Args:
            user_id: 用户ID
            content: 记忆内容
            memory_type: 记忆类型（general/medical/conversation等）

        Returns:
            是否保存成功
        """
        if not self._ensure_collection():
            return False

        from pymilvus import Collection, connections

        try:
            # 文本向量化
            embedding = self.embedding_model.embed_query(content)

            # 插入数据
            collection = Collection(self.COLLECTION_MEMORY, using="memory")
            collection.insert([
                [user_id],
                [content],
                [memory_type],
                [embedding],
            ])
            collection.flush()

            return True

        except Exception:
            return False
        finally:
            try:
                connections.disconnect("memory")
            except Exception:
                pass

    def search_memory(
        self,
        user_id: str,
        query: str = "",
        top_k: int = 5,
        memory_type: Optional[str] = None,
    ) -> list[dict]:
        """从Milvus中检索用户记忆

        Args:
            user_id: 用户ID
            query: 查询文本，为空时使用零向量返回最新记忆
            top_k: 返回的记忆数量
            memory_type: 过滤记忆类型，为None时不限制

        Returns:
            记忆列表，每项包含 content, memory_type, score
        """
        if not self._ensure_collection():
            return []

        from pymilvus import Collection, connections

        try:
            collection = Collection(self.COLLECTION_MEMORY, using="memory")
            collection.load()

            # 生成查询向量
            if query:
                query_vector = self.embedding_model.embed_query(query)
            else:
                query_vector = [0.0] * self.EMBEDDING_DIM

            # 构建过滤表达式
            expr = f'user_id == "{user_id}"'
            if memory_type:
                expr += f' && memory_type == "{memory_type}"'

            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10},
            }

            results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["content", "memory_type"],
            )

            memories = []
            if results and len(results) > 0:
                for hit in results[0]:
                    memories.append({
                        "content": hit.entity.get("content", ""),
                        "memory_type": hit.entity.get("memory_type", ""),
                        "score": round(hit.score, 4),
                    })

            return memories

        except Exception:
            return []
        finally:
            try:
                connections.disconnect("memory")
            except Exception:
                pass
