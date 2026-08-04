# -*- coding: utf-8 -*-
"""Milvus 向量索引初始化 - 为症状名称创建向量嵌入"""
import os
import sys

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from medical_agent.core.config import get_settings
from medical_agent.providers.embedding import get_embedding_model
from neo4j import GraphDatabase

# 集合名称
COLLECTION_NAME = "symptoms"
# 向量维度
EMBEDDING_DIM = 1024
# 批次大小
BATCH_SIZE = 100
# Milvus 连接别名
MILVUS_ALIAS = "init_milvus"


def connect_milvus(settings) -> None:
    """连接 Milvus 服务"""
    connections.connect(
        alias=MILVUS_ALIAS,
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
    )
    print(f"   >> 连接 Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")


def create_collection() -> Collection:
    """创建症状向量集合（如已存在则先删除）"""
    if utility.has_collection(COLLECTION_NAME, using=MILVUS_ALIAS):
        utility.drop_collection(COLLECTION_NAME, using=MILVUS_ALIAS)
        print(f"   >> 已删除旧集合 {COLLECTION_NAME}")

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
            description="自增主键",
        ),
        FieldSchema(
            name="name",
            dtype=DataType.VARCHAR,
            max_length=200,
            description="症状名称",
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM,
            description="症状名称向量嵌入",
        ),
    ]
    schema = CollectionSchema(fields, description="医学症状向量集合")
    collection = Collection(COLLECTION_NAME, schema, using=MILVUS_ALIAS)
    print(f"   >> 创建集合 {COLLECTION_NAME}(dim={EMBEDDING_DIM})")

    # 创建 COSINE 相似度索引
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("   >> 创建 COSINE+IVF_FLAT 索引")

    return collection


def get_symptoms_from_neo4j(settings) -> list[str]:
    """从 Neo4j 查询所有症状节点名称"""
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run("MATCH (s:Symptom) RETURN s.name AS name ORDER BY s.name")
            symptoms = [record["name"] for record in result]
        print(f"   >> 从 Neo4j 获取 {len(symptoms)} 个症状名称")
        return symptoms
    finally:
        driver.close()


def embed_and_insert(collection: Collection, symptoms: list[str]) -> int:
    """批量嵌入并插入 Milvus"""
    embedding_model = get_embedding_model()
    print(f"   >> 使用 Embedding 模型: {embedding_model.model}")

    total = 0
    for i in range(0, len(symptoms), BATCH_SIZE):
        batch = symptoms[i : i + BATCH_SIZE]
        vectors = embedding_model.embed_documents(batch)
        entities = [
            {"name": name, "embedding": vector}
            for name, vector in zip(batch, vectors)
        ]
        collection.insert(entities)
        total += len(batch)
        print(f"   >> 进度: {total}/{len(symptoms)}")

    collection.flush()
    print(f"   >> 全部插入完成，共 {total} 条")
    return total


def main():
    """主入口：创建 Milvus 症状向量索引"""
    settings = get_settings()

    # 连接 Milvus
    connect_milvus(settings)

    try:
        # 创建集合
        print(f"\n[1/3] 创建集合 {COLLECTION_NAME}...")
        collection = create_collection()

        # 从 Neo4j 获取症状
        print("\n[2/3] 从 Neo4j 获取症状数据...")
        symptoms = get_symptoms_from_neo4j(settings)

        if not symptoms:
            print("   >> 警告: Neo4j 中无症状数据，请先运行 init_neo4j.py")
            return

        # 嵌入并插入
        print(f"\n[3/3] 嵌入并插入 {len(symptoms)} 条症状向量...")
        count = embed_and_insert(collection, symptoms)

        # 加载集合到内存
        collection.load()
        print(f"\n{'=' * 50}")
        print(f"Milvus 症状向量索引构建完成！")
        print(f"  集合: {COLLECTION_NAME}")
        print(f"  记录数: {count}")
        print(f"  向量维度: {EMBEDDING_DIM}")
        print("=" * 50)

    finally:
        connections.disconnect(MILVUS_ALIAS)
        print("\nMilvus 连接已关闭")


if __name__ == "__main__":
    main()
