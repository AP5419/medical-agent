# -*- coding: utf-8 -*-
"""Embedding提供层 - 文本向量化接口"""

from langchain_community.embeddings import DashScopeEmbeddings

from medical_agent.core.config import get_settings


def get_embedding_model() -> DashScopeEmbeddings:
    """获取DashScope Embedding模型实例"""
    settings = get_settings()
    return DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )
