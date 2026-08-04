# -*- coding: utf-8 -*-
"""医疗RAG引擎 - 基于Milvus向量检索的医学知识问答"""

import json

from medical_agent.core.config import get_settings
from medical_agent.providers.embedding import get_embedding_model
from medical_agent.providers.llm import get_llm_qa


class MedicalRAGEngine:
    """医疗领域RAG检索增强生成引擎，支持查询重写、HyDE、向量检索、重排序等功能"""

    def __init__(self):
        """初始化RAG引擎，加载嵌入模型和配置"""
        self.embedding_model = get_embedding_model()
        self.llm = get_llm_qa()
        self.settings = get_settings()

    def rewrite_query(self, query: str) -> str:
        """使用LLM将口语化表达转换为医学专业术语"""
        prompt = f"""你是一个医学查询重写助手。请将以下口语化问题改写为专业的医学查询语句，
使用标准医学术语，保留原始含义。只返回改写后的查询文本，不要额外解释。

原始查询：{query}

改写后的查询："""
        response = self.llm.invoke(prompt)
        return response.content.strip() if hasattr(response, "content") else str(response).strip()

    def generate_hyde_document(self, query: str) -> str:
        """生成假设性答案文档(HyDE)，用于提高向量检索的召回率"""
        prompt = f"""你是一个医学科普助手。请根据以下问题，生成一篇简短的医学知识短文作为假设答案，
涵盖可能相关的知识点。只返回短文内容，不要额外解释。

问题：{query}

假设答案短文："""
        response = self.llm.invoke(prompt)
        return response.content.strip() if hasattr(response, "content") else str(response).strip()

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_hyde: bool = True,
        use_rewrite: bool = True,
    ) -> list[dict]:
        """完整的RAG检索流程：查询重写 → HyDE生成 → 向量嵌入 → Milvus检索 → 重排序

        Args:
            query: 用户原始查询
            top_k: 返回的文档数量
            use_hyde: 是否启用HyDE增强检索
            use_rewrite: 是否启用查询重写

        Returns:
            格式化后的检索结果列表，每项包含 content, source, title, score
        """
        from medical_agent.infra.milvus import COLLECTION_DOCS, get_collection

        processed_query = query

        # 步骤1：查询重写
        if use_rewrite:
            try:
                processed_query = self.rewrite_query(query)
            except Exception:
                processed_query = query

        # 步骤2：HyDE生成假设文档
        search_text = processed_query
        if use_hyde:
            try:
                hyde_doc = self.generate_hyde_document(processed_query)
                search_text = f"{processed_query}\n{hyde_doc}"
            except Exception:
                pass

        # 步骤3：文本向量化
        try:
            query_vector = self.embedding_model.embed_query(search_text)
        except Exception as e:
            return [{"content": f"向量化失败: {e}", "source": "", "title": "向量化错误", "score": 0.0}]

        # 步骤4：Milvus向量检索
        try:
            collection = get_collection(COLLECTION_DOCS)
            if collection is None:
                return [{"content": "知识库集合不存在，请先导入文档", "source": "", "title": "集合未找到", "score": 0.0}]

            collection.load()

            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k * 2,
                output_fields=["content", "source", "title"],
            )

            if not results or len(results[0]) == 0:
                return [{"content": "未找到相关知识文档", "source": "", "title": "无结果", "score": 0.0}]

            # 步骤5：按分数排序并格式化结果
            hits = sorted(results[0], key=lambda x: x.score, reverse=True)
            formatted_results = []
            for hit in hits[:top_k]:
                formatted_results.append({
                    "content": hit.entity.get("content", ""),
                    "source": hit.entity.get("source", ""),
                    "title": hit.entity.get("title", ""),
                    "score": round(hit.score, 4),
                })

            return formatted_results

        except Exception as e:
            return [{"content": f"检索异常: {e}", "source": "", "title": "检索错误", "score": 0.0}]

    def generate_answer(self, query: str, documents: list[dict]) -> tuple[str, list[str]]:
        """基于检索到的文档生成医学答案，并附带引用来源

        Args:
            query: 用户原始问题
            documents: 检索到的相关文档列表

        Returns:
            (answer_text, sources_list) 包含来源标注的答案和引用列表
        """
        if not documents or all(not d.get("content") for d in documents):
            return "未找到相关信息，无法回答该医学问题。", []

        # 构建文档上下文
        context_parts = []
        sources = []
        for i, doc in enumerate(documents, 1):
            title = doc.get("title", f"文档{i}")
            source = doc.get("source", "未知来源")
            content = doc.get("content", "")
            context_parts.append(f"[文档{i}] {title}\n来源：{source}\n内容：{content}")
            sources.append(title)

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""你是一个专业的医学AI助手。请基于以下参考文档回答用户问题。
回答应专业、准确、易懂，并尽可能引用文档中的具体信息。

参考文档：
{context}

用户问题：{query}

请在回答末尾附上参考来源标记，格式为：【参考来源: 文档1, 文档2】

请回答："""
        response = self.llm.invoke(prompt)
        answer = response.content.strip() if hasattr(response, "content") else str(response).strip()

        return answer, sources
