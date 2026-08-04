# -*- coding: utf-8 -*-
"""
Layer 5: 能力引擎层
模块: MS GraphRAG 引擎封装
技术栈: microsoft/graphrag 官方包
职责: 非结构化医疗文档的 Local/Global Search
"""

import os
from pathlib import Path
from typing import Optional

from loguru import logger


class MSGraphRAGEngine:
    """
    Microsoft GraphRAG 引擎封装
    
    基于官方 graphrag 包，提供 Local Search 和 Global Search 两种查询模式。
    处理药品说明书、临床指南、医学科普等非结构化文档。
    
    使用方式:
        engine = MSGraphRAGEngine()
        result = await engine.local_search("二甲双胍的常见副作用是什么？")
        result = await engine.global_search("这些文档中反复出现的高血压用药模式是什么？")
    """
    
    def __init__(self, root_dir: Optional[str] = None):
        """
        初始化 MS GraphRAG 引擎
        
        Args:
            root_dir: 项目根目录（包含 settings.yaml 和 output/ 的目录）
        """
        if root_dir is None:
            root_dir = str(Path(__file__).resolve().parent.parent.parent.parent.parent)
        self.root_dir = root_dir
        self._initialized = False
        self._search_engine = None
        self._context_builder = None
        logger.info(f"MS GraphRAG 引擎已初始化，根目录: {self.root_dir}")
    
    def _ensure_initialized(self):
        """延迟加载 graphrag 查询引擎"""
        if self._initialized:
            return
        
        try:
            from graphrag.config.create_graphrag_config import create_graphrag_config
            from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
            from graphrag.query.indexer_adapters import (
                read_indexer_entities,
                read_indexer_relationships,
                read_indexer_reports,
                read_indexer_text_units,
            )
            from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
            from graphrag.query.llm.oai.chat_openai import ChatOpenAI
            from graphrag.query.llm.oai.embedding import OpenAIEmbedding
            from graphrag.query.llm.oai.typing import OpenaiApiType
            from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
            from graphrag.query.structured_search.local_search.search import LocalSearch
            from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
            from graphrag.query.structured_search.global_search.search import GlobalSearch
            from graphrag.vector_stores.lancedb import LanceDBVectorStore
            
            import yaml
            
            # 加载配置
            config_path = Path(self.root_dir) / "settings.yaml"
            if not config_path.exists():
                logger.error(f"settings.yaml 不存在: {config_path}")
                return
            
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            
            # 构建配置
            config = create_graphrag_config(config_data, self.root_dir)
            
            # 读取索引数据
            output_dir = Path(self.root_dir) / "output"
            entities = read_indexer_entities(
                output_dir / "create_final_entities.parquet",
                output_dir / "create_final_nodes.parquet",
                config.community_level,
            )
            relationships = read_indexer_relationships(output_dir / "create_final_relationships.parquet")
            reports = read_indexer_reports(
                output_dir / "create_final_community_reports.parquet",
                output_dir / "create_final_communities.parquet",
                config.community_level,
            )
            text_units = read_indexer_text_units(output_dir / "create_final_text_units.parquet")
            
            # 实体向量存储
            description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")
            description_embedding_store.connect(
                db_uri=output_dir / "lancedb",
            )
            entities = store_entity_semantic_embeddings(
                entities=entities,
                vectorstore=description_embedding_store,
            )
            
            # LLM 实例
            llm = ChatOpenAI(
                api_key=os.environ.get("GRAPHRAG_API_KEY", ""),
                api_base=os.environ.get("GRAPHRAG_API_BASE", "https://api.deepseek.com"),
                model=config.llm.model,
                api_type=OpenaiApiType.OpenAI,
                max_tokens=config.llm.max_tokens,
                temperature=config.llm.temperature,
            )
            
            text_embedder = OpenAIEmbedding(
                api_key=os.environ.get("GRAPHRAG_API_KEY", ""),
                api_base=os.environ.get("GRAPHRAG_API_BASE", "https://api.deepseek.com"),
                model=config.embeddings.llm.model,
                api_type=OpenaiApiType.OpenAI,
            )
            
            # Local Search 上下文构建器
            local_context_builder = LocalSearchMixedContext(
                community_reports=reports,
                text_units=text_units,
                entities=entities,
                relationships=relationships,
                entity_text_embeddings=description_embedding_store,
                embedding_vectorstore_key=EntityVectorStoreKey.ID,
                text_embedder=text_embedder,
                token_encoder=config.get_text_embedder(),
            )
            
            # 初始化 Local Search
            self._local_search = LocalSearch(
                model=llm,
                context_builder=local_context_builder,
                token_encoder=config.get_text_embedder(),
                system_prompt=(
                    "你是一个医疗知识助手。基于提供的文档内容回答用户问题。"
                    "回答应专业、准确，引用具体文档中的信息。"
                ),
                response_type="医学专业回答，包含引用来源",
                max_data_tokens=config.local_search.max_data_tokens,
            )
            
            # 初始化 Global Search
            global_context_builder = GlobalCommunityContext(
                community_reports=reports,
                entities=entities,
                token_encoder=config.get_text_embedder(),
            )
            
            self._global_search = GlobalSearch(
                model=llm,
                context_builder=global_context_builder,
                token_encoder=config.get_text_embedder(),
                max_data_tokens=config.global_search.max_data_tokens,
                map_max_tokens=config.global_search.map_max_tokens,
                reduce_max_tokens=config.global_search.reduce_max_tokens,
                response_type="医学综合分析报告",
            )
            
            self._initialized = True
            logger.info("MS GraphRAG 查询引擎初始化完成")
            
        except ImportError as e:
            logger.error(f"MS GraphRAG 未安装或版本不兼容: {e}")
            self._search_engine = None
        except FileNotFoundError as e:
            logger.error(f"索引数据不存在，请先运行 graphrag index: {e}")
            self._search_engine = None
        except Exception as e:
            logger.error(f"MS GraphRAG 初始化失败: {e}")
            self._search_engine = None
    
    async def local_search(self, query: str) -> dict:
        """
        Local Search: 实体级精确查询
        
        适用于: "二甲双胍的副作用是什么？" "高血压的诊断标准是什么？"
        原理: 识别查询中的实体 → 提取相关文本片段/关系/社区报告 → 生成回答
        
        Args:
            query: 用户查询
            
        Returns:
            {"answer": str, "sources": list[str], "context": dict}
        """
        self._ensure_initialized()
        
        if self._local_search is None:
            return {"answer": "MS GraphRAG 引擎未初始化，请先运行 graphrag index", "sources": [], "context": {}}
        
        try:
            result = await self._local_search.asearch(query)
            return {
                "answer": result.response,
                "sources": result.context_data.get("reports", []),
                "context": result.context_data,
            }
        except Exception as e:
            logger.error(f"Local Search 失败: {e}")
            return {"answer": f"查询失败: {str(e)}", "sources": [], "context": {}}
    
    async def global_search(self, query: str) -> dict:
        """
        Global Search: 全数据集综合查询
        
        适用于: "这些文档中高血压治疗的主要模式是什么？"
        原理: Map-Reduce 模式 → 分批处理社区报告 → 汇总答案
        
        Args:
            query: 用户查询
            
        Returns:
            {"answer": str, "context": dict}
        """
        self._ensure_initialized()
        
        if self._global_search is None:
            return {"answer": "MS GraphRAG 引擎未初始化，请先运行 graphrag index", "context": {}}
        
        try:
            result = await self._global_search.asearch(query)
            return {
                "answer": result.response,
                "context": result.context_data,
            }
        except Exception as e:
            logger.error(f"Global Search 失败: {e}")
            return {"answer": f"查询失败: {str(e)}", "context": {}}
